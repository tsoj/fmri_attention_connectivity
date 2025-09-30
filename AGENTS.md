This is a research project: assume the happy path and fail fast. If something unexpected happens, abort with a clear stacktrace; catching exceptions happens only when adding context and re-raising is truly necessary.

To avoid asserts being optimize away, use `check_assert_enabled()` (`from utils import check_assert_enabled`) at the beginning of every main entry point.

Only have main entry points in modules that are necessary. Avoid library style files that also have a main function / entrypoint.

Configuration is explicit. Important settings are passed as arguments or assembled once at the entrypoint (e.g., CLI) and then threaded through; avoid hidden defaults, global state, and magic environment lookups.

Typing is pragmatic. Add type hints when they aid clarity and safety—especially for public functions, return values, and data structures—but don’t contort code just to satisfy a checker.

Make assumptions explicit with assert for invariants and impossible branches; treat assertion failures as bugs. Never use bare except; only catch to add context and then raise from.

Avoid duplication. Extract shared logic and keep functions small and single-purpose; if two implementations must diverge, document why.

Outputs are reproducible. Each run writes to a new results directory, never overwrites, and records config, seeds (if any), package versions, and the Git commit alongside artifacts.

Prefer the modern standard library.

Use idiomatic, modern Python and keep the codebase straightforward to read, run, and reproduce
