"""Code generation module for producing actual code files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog

from src.core.config import Settings
from src.core.models import Answer, CodeArtifact, Question, QuestionType
from src.utils.ai_client import AIClient, get_ai_client

logger = structlog.get_logger(__name__)


class CodeGenerator:
    """
    Generates production-ready code from Q&A pairs.

    Features:
    - Multi-language support
    - Project structure awareness
    - Code quality validation
    - Test generation
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the code generator."""
        self.settings = settings
        self._ai_client: AIClient | None = None

        # Language-specific configurations
        self._language_configs = self._load_language_configs()

    @property
    def ai_client(self) -> AIClient:
        """Get the AI client (lazy init)."""
        if self._ai_client is None:
            self._ai_client = get_ai_client(self.settings)
        return self._ai_client

    async def generate(
        self,
        question: Question,
        answer: Answer,
        context: dict[str, Any],
        existing_artifacts: list[CodeArtifact] | None = None,
    ) -> CodeArtifact:
        """
        Generate a code artifact from a Q&A pair.

        Args:
            question: The question that prompted this code
            answer: The answer containing implementation details
            context: Current project context
            existing_artifacts: Previously generated artifacts for context

        Returns:
            Generated code artifact
        """
        logger.info(
            "Generating code artifact",
            question_id=question.id,
            question_type=question.question_type,
        )

        # Determine target language and file path
        language = self._determine_language(context, answer)
        file_path = self._determine_file_path(question, context, language)

        # Check if we should update existing artifact or create new
        existing = self._find_existing_artifact(file_path, existing_artifacts or [])

        if existing:
            # Update existing code
            code_content = await self._update_code(
                existing=existing,
                question=question,
                answer=answer,
                context=context,
            )
            version = existing.version + 1
        else:
            # Generate new code
            code_content = await self._generate_new_code(
                question=question,
                answer=answer,
                context=context,
                language=language,
                file_path=file_path,
                existing_artifacts=existing_artifacts,
            )
            version = 1

        # Validate the generated code
        is_valid, lint_errors = await self._validate_code(code_content, language)

        # Create the artifact
        artifact = CodeArtifact(
            session_id=context.get("session_id", "unknown"),
            file_path=file_path,
            content=code_content,
            language=language,
            question_id=question.id,
            version=version,
            is_valid=is_valid,
            lint_errors=lint_errors,
        )

        logger.info(
            "Code artifact generated",
            file_path=file_path,
            language=language,
            is_valid=is_valid,
            lines=len(code_content.split("\n")),
        )

        return artifact

    async def generate_tests(
        self,
        artifact: CodeArtifact,
        context: dict[str, Any],
    ) -> CodeArtifact:
        """
        Generate tests for a code artifact.

        Args:
            artifact: The code artifact to test
            context: Current project context

        Returns:
            Test code artifact
        """
        prompt = f"""Generate comprehensive tests for the following code:

File: {artifact.file_path}
Language: {artifact.language}

```{artifact.language}
{artifact.content}
```

Requirements:
1. Write unit tests covering all public functions/methods
2. Include edge cases and error conditions
3. Use appropriate testing framework for the language
4. Add descriptive test names
5. Include setup and teardown if needed

Generate the test file content:
"""

        response = await self.ai_client.generate(
            prompt=prompt,
            system_prompt=self._get_test_system_prompt(artifact.language),
            max_tokens=4000,
        )

        # Extract test code
        test_code = self._extract_code_content(response, artifact.language)

        # Determine test file path
        test_path = self._get_test_file_path(artifact.file_path, artifact.language)

        return CodeArtifact(
            session_id=artifact.session_id,
            file_path=test_path,
            content=test_code,
            language=artifact.language,
            question_id=artifact.question_id,
            version=1,
            metadata={"type": "test", "tests_for": artifact.file_path},
        )

    async def _generate_new_code(
        self,
        question: Question,
        answer: Answer,
        context: dict[str, Any],
        language: str,
        file_path: str,
        existing_artifacts: list[CodeArtifact] | None,
    ) -> str:
        """Generate new code from scratch."""
        prompt = self._build_generation_prompt(
            question=question,
            answer=answer,
            context=context,
            language=language,
            file_path=file_path,
            existing_artifacts=existing_artifacts,
        )

        response = await self.ai_client.generate(
            prompt=prompt,
            system_prompt=self._get_code_system_prompt(language),
            max_tokens=6000,
        )

        return self._extract_code_content(response, language)

    async def _update_code(
        self,
        existing: CodeArtifact,
        question: Question,
        answer: Answer,
        context: dict[str, Any],
    ) -> str:
        """Update existing code based on new requirements."""
        prompt = f"""Update the following code based on the new requirements:

Current Code ({existing.file_path}):
```{existing.language}
{existing.content}
```

Question: {question.content}

New Requirements:
{answer.content[:2000]}

Instructions:
1. Modify the existing code to address the requirements
2. Maintain backward compatibility where possible
3. Keep the code style consistent
4. Add any new necessary imports
5. Update comments/docstrings as needed

Generate the updated code:
"""

        response = await self.ai_client.generate(
            prompt=prompt,
            system_prompt=self._get_code_system_prompt(existing.language),
            max_tokens=6000,
        )

        return self._extract_code_content(response, existing.language)

    def _build_generation_prompt(
        self,
        question: Question,
        answer: Answer,
        context: dict[str, Any],
        language: str,
        file_path: str,
        existing_artifacts: list[CodeArtifact] | None,
    ) -> str:
        """Build the prompt for code generation."""
        parts = []

        # Project context
        project = context.get("project", {})
        if project:
            parts.append(f"## Project: {project.get('name', 'Unknown')}")
            parts.append(f"Architecture: {project.get('architecture_type', 'standard')}")
            parts.append("")

        # Existing codebase context
        if existing_artifacts:
            parts.append("## Existing Code Context")
            for artifact in existing_artifacts[-5:]:
                parts.append(f"- {artifact.file_path} ({artifact.language})")
            parts.append("")

        # Requirements from Q&A
        parts.append("## Requirements")
        parts.append(f"Question: {question.content}")
        parts.append("")
        parts.append("Answer/Specification:")
        parts.append(answer.content[:3000])
        parts.append("")

        # Code from answer if available
        if answer.code_snippets:
            parts.append("## Reference Code Snippets")
            for i, snippet in enumerate(answer.code_snippets[:3], 1):
                parts.append(f"Snippet {i}:")
                parts.append(f"```{language}")
                parts.append(snippet[:1000])
                parts.append("```")
            parts.append("")

        # Generation instructions
        parts.append("## Generation Instructions")
        parts.append(f"Target File: {file_path}")
        parts.append(f"Language: {language}")
        parts.append("")

        lang_config = self._language_configs.get(language, {})
        parts.append("Requirements:")
        parts.append("1. Generate complete, production-ready code")
        parts.append("2. Include all necessary imports")
        parts.append("3. Add proper error handling")
        parts.append("4. Include type hints/annotations")
        parts.append("5. Add documentation/docstrings")
        parts.append(f"6. Follow {lang_config.get('style_guide', 'standard')} style guide")
        parts.append("")
        parts.append("Generate the code:")

        return "\n".join(parts)

    def _get_code_system_prompt(self, language: str) -> str:
        """Get system prompt for code generation."""
        lang_config = self._language_configs.get(language, {})

        return f"""You are an expert {language} developer creating production-ready code.

Guidelines:
1. Write clean, maintainable, and well-documented code
2. Follow {lang_config.get('style_guide', 'standard')} style conventions
3. Include comprehensive error handling
4. Use appropriate design patterns
5. Add type hints/annotations where applicable
6. Include docstrings/comments explaining complex logic
7. Consider edge cases and input validation
8. Make the code testable

{lang_config.get('additional_guidelines', '')}

Output only the code, wrapped in appropriate markdown code blocks.
"""

    def _get_test_system_prompt(self, language: str) -> str:
        """Get system prompt for test generation."""
        test_frameworks = {
            "python": "pytest",
            "typescript": "jest",
            "javascript": "jest",
            "go": "testing package",
            "rust": "cargo test",
            "java": "JUnit 5",
        }

        framework = test_frameworks.get(language, "standard testing framework")

        return f"""You are an expert {language} developer writing comprehensive tests.

Use {framework} for testing.

Guidelines:
1. Write clear, descriptive test names
2. Test all public functions/methods
3. Include edge cases and error conditions
4. Use appropriate assertions
5. Mock external dependencies
6. Follow Arrange-Act-Assert pattern
7. Aim for high code coverage
8. Test both success and failure cases

Output only the test code, wrapped in appropriate markdown code blocks.
"""

    def _determine_language(self, context: dict[str, Any], answer: Answer) -> str:
        """Determine the target programming language."""
        # Check context for project language
        project = context.get("project", {})
        target_languages = project.get("target_languages", [])

        if target_languages:
            return target_languages[0]

        # Check answer code snippets for language hints
        if answer.code_snippets:
            for snippet in answer.code_snippets:
                detected = self._detect_language(snippet)
                if detected:
                    return detected

        # Default to Python
        return "python"

    def _detect_language(self, code: str) -> str | None:
        """Detect programming language from code snippet."""
        indicators = {
            "python": ["def ", "import ", "from ", "class ", "self.", "async def"],
            "typescript": ["interface ", "type ", ": string", ": number", "export "],
            "javascript": ["const ", "let ", "function ", "=>", "require("],
            "go": ["func ", "package ", "import (", "type struct"],
            "rust": ["fn ", "let mut", "impl ", "pub fn", "use "],
            "java": ["public class", "private ", "void ", "System.out"],
        }

        code_lower = code.lower()
        scores: dict[str, int] = {}

        for lang, patterns in indicators.items():
            scores[lang] = sum(1 for p in patterns if p.lower() in code_lower)

        if scores:
            best_lang = max(scores, key=lambda k: scores[k])
            if scores[best_lang] > 0:
                return best_lang

        return None

    def _determine_file_path(
        self,
        question: Question,
        context: dict[str, Any],
        language: str,
    ) -> str:
        """Determine the file path for generated code."""
        project = context.get("project", {})
        component = question.context.get("component", "")

        # Language-specific extensions and directories
        lang_config = self._language_configs.get(language, {})
        extension = lang_config.get("extension", ".txt")
        src_dir = lang_config.get("src_dir", "src")

        if component:
            # Use component name for file
            file_name = self._to_file_name(component, language)
        else:
            # Generate from question
            file_name = self._generate_file_name(question.content, language)

        return f"{src_dir}/{file_name}{extension}"

    def _to_file_name(self, component: str, language: str) -> str:
        """Convert component name to file name."""
        # Convert to appropriate case based on language
        name = component.lower().replace(" ", "_").replace("-", "_")

        # Remove non-alphanumeric characters
        name = re.sub(r"[^a-z0-9_]", "", name)

        if language in ["typescript", "javascript"]:
            # Convert to camelCase for JS/TS
            parts = name.split("_")
            name = parts[0] + "".join(p.capitalize() for p in parts[1:])

        return name

    def _generate_file_name(self, question: str, language: str) -> str:
        """Generate file name from question content."""
        # Extract key words
        words = re.findall(r"\b\w+\b", question.lower())
        keywords = [w for w in words if len(w) > 3 and w not in {"how", "what", "should", "the", "this", "that", "with", "from"}]

        if keywords:
            name = "_".join(keywords[:3])
        else:
            name = "generated"

        return self._to_file_name(name, language)

    def _get_test_file_path(self, source_path: str, language: str) -> str:
        """Generate test file path from source path."""
        path = Path(source_path)
        name = path.stem

        test_patterns = {
            "python": f"tests/test_{name}.py",
            "typescript": f"tests/{name}.test.ts",
            "javascript": f"tests/{name}.test.js",
            "go": f"{path.parent}/{name}_test.go",
            "rust": f"{path.parent}/{name}_test.rs",
            "java": f"src/test/java/{name}Test.java",
        }

        return test_patterns.get(language, f"tests/test_{name}.txt")

    def _find_existing_artifact(
        self, file_path: str, artifacts: list[CodeArtifact]
    ) -> CodeArtifact | None:
        """Find existing artifact with same file path."""
        for artifact in artifacts:
            if artifact.file_path == file_path:
                return artifact
        return None

    def _extract_code_content(self, response: str, language: str) -> str:
        """Extract code content from AI response."""
        # Try to find code blocks with language tag
        pattern = rf"```{language}\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            return matches[0].strip()

        # Try generic code blocks
        pattern = r"```\w*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            # Return the longest code block
            return max(matches, key=len).strip()

        # If no code blocks, return the response as-is (might be just code)
        # Remove any markdown formatting
        response = re.sub(r"^```\w*\n?", "", response)
        response = re.sub(r"\n?```$", "", response)

        return response.strip()

    async def _validate_code(
        self, code: str, language: str
    ) -> tuple[bool, list[str]]:
        """
        Validate generated code.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: list[str] = []

        # Basic validation
        if not code.strip():
            errors.append("Empty code generated")
            return False, errors

        # Language-specific validation
        lang_config = self._language_configs.get(language, {})
        required_patterns = lang_config.get("required_patterns", [])

        for pattern in required_patterns:
            if not re.search(pattern, code):
                errors.append(f"Missing required pattern: {pattern}")

        # Check for common issues
        if "TODO" in code or "FIXME" in code:
            errors.append("Contains TODO/FIXME markers")

        if "..." in code and language != "python":  # Ellipsis ok in Python
            errors.append("Contains placeholder ellipsis")

        # Syntax validation would go here (using language-specific tools)
        # For now, we do basic checks

        is_valid = len(errors) == 0
        return is_valid, errors

    def _load_language_configs(self) -> dict[str, dict[str, Any]]:
        """Load language-specific configurations."""
        return {
            "python": {
                "extension": ".py",
                "src_dir": "src",
                "style_guide": "PEP 8",
                "required_patterns": [],
                "additional_guidelines": """
- Use type hints for all function parameters and returns
- Use docstrings following Google style
- Prefer f-strings for string formatting
- Use pathlib for file paths
""",
            },
            "typescript": {
                "extension": ".ts",
                "src_dir": "src",
                "style_guide": "TypeScript standard",
                "required_patterns": [],
                "additional_guidelines": """
- Use strict mode
- Define interfaces for all data structures
- Use async/await for asynchronous code
- Export types alongside implementations
""",
            },
            "javascript": {
                "extension": ".js",
                "src_dir": "src",
                "style_guide": "ESLint standard",
                "required_patterns": [],
                "additional_guidelines": """
- Use const by default, let when needed
- Use arrow functions for callbacks
- Use async/await for asynchronous code
- Add JSDoc comments for documentation
""",
            },
            "go": {
                "extension": ".go",
                "src_dir": "pkg",
                "style_guide": "Go standard",
                "required_patterns": [r"^package "],
                "additional_guidelines": """
- Follow effective Go guidelines
- Use meaningful variable names
- Handle all errors explicitly
- Add comments for exported functions
""",
            },
            "rust": {
                "extension": ".rs",
                "src_dir": "src",
                "style_guide": "Rust standard",
                "required_patterns": [],
                "additional_guidelines": """
- Use Result for error handling
- Implement traits where appropriate
- Use lifetimes explicitly when needed
- Add documentation comments with ///
""",
            },
            "java": {
                "extension": ".java",
                "src_dir": "src/main/java",
                "style_guide": "Google Java Style",
                "required_patterns": [r"public class"],
                "additional_guidelines": """
- Use appropriate access modifiers
- Follow SOLID principles
- Add Javadoc for public methods
- Use Optional for nullable returns
""",
            },
        }
