"""Tool schemas for LCM — what the LLM sees."""

LCM_GREP = {
    "name": "lcm_grep",
    "description": (
        "Search the plugin-local LCM database for past conversation content. "
        "Default scope is the active session and returns both raw messages and summary nodes across all depths. "
        "Broader scopes ('all' or 'session') must be requested explicitly and exist for bounded archive recovery "
        "over rows already present in lcm.db, including externally backfilled rows that may carry source strings "
        "such as openclaw-lcm:* . In broader scopes only raw-message hits are returned; cross-session summary "
        "node expansion is intentionally deferred. Use lcm_expand(store_id=...) on a cross-session message hit "
        "to drill into its full content. For Hermes-tracked session history outside the LCM database, use session_search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query (FTS5 syntax: keywords, phrases, OR/NOT). "
                    "FTS5 defaults to AND matching, so prefer 1-3 distinctive terms or one quoted multi-word phrase. "
                    "Wrap exact phrases in quotes. Short CJK fragments and emoji-heavy queries may use substring fallback instead of plain FTS token matching."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Max results to return (default 10, hard upper bound 200). "
                    "Values above the cap are clamped and reported via limit_clamped_from in the response."
                ),
                "default": 10,
            },
            "sort": {
                "type": "string",
                "enum": ["recency", "relevance", "hybrid"],
                "description": (
                    "How to order matches. 'recency' favors newer hits, 'relevance' favors strongest FTS matches, "
                    "and 'hybrid' keeps strong older matches competitive while still boosting newer context."
                ),
                "default": "recency",
            },
            "session_scope": {
                "type": "string",
                "enum": ["current", "all", "session"],
                "description": (
                    "Scope of the search across the plugin-local LCM database. "
                    "'current' (default) restricts to the active session and preserves historical behavior. "
                    "'all' searches every session in the local LCM database. "
                    "'session' restricts to the session_id supplied via the session_id parameter. "
                    "Cross-session search returns snippets and message store_ids; cross-session summary node expansion is deferred. "
                    "For Hermes-tracked session history outside the LCM database, use session_search."
                ),
                "default": "current",
            },
            "session_id": {
                "type": "string",
                "description": (
                    "When session_scope='session', the explicit session id to restrict the search to. "
                    "Must not be supplied with session_scope='current' or session_scope='all'."
                ),
            },
            "source": {
                "type": "string",
                "description": (
                    "Optional source/platform filter (for example cli, discord, telegram). "
                    "Applies directly to raw messages and to summaries via descendant source lineage. "
                    "Use 'unknown' for explicit unknown-source content."
                ),
            },
            "role": {
                "type": "string",
                "enum": ["system", "user", "assistant", "tool", "unknown"],
                "description": "Optional raw-message role filter. When supplied, lcm_grep returns raw message hits only.",
            },
            "time_from": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Optional inclusive minimum raw-message timestamp. Accepts Unix seconds or timezone-aware ISO 8601; "
                    "naive ISO timestamps are rejected. When supplied, lcm_grep returns raw message hits only."
                ),
            },
            "time_to": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Optional inclusive maximum raw-message timestamp. Accepts Unix seconds or timezone-aware ISO 8601; "
                    "naive ISO timestamps are rejected. When supplied, lcm_grep returns raw message hits only."
                ),
            },
        },
        "required": ["query"],
    },
}

LCM_LOAD_SESSION = {
    "name": "lcm_load_session",
    "description": (
        "Load an ordered raw-message transcript page for one explicit session_id from the plugin-local LCM database. "
        "This is enumeration, not search: it does not require a query, returns raw message content rather than snippets, "
        "and orders rows chronologically by store_id. Use this after session_search or lcm_grep has identified a session_id "
        "that already exists in lcm.db. Output is bounded by limit, per-row content is bounded by max_content_chars, "
        "and row pagination uses after_store_id/next_cursor. "
        "It returns raw rows only; cross-session summary/DAG expansion remains out of scope."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Explicit LCM session id to load. Required; no implicit current/all fallback is applied.",
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum raw messages to return (default 100, hard upper bound 200). "
                    "Values above the cap are clamped and reported via limit_clamped_from."
                ),
                "default": 100,
            },
            "max_content_chars": {
                "type": "integer",
                "description": (
                    "Maximum content characters to include per message (default 4000, hard upper bound 20000). "
                    "Longer rows include content_truncated=true and can be recovered fully with lcm_expand(store_id=...)."
                ),
                "default": 4000,
            },
            "after_store_id": {
                "type": "integer",
                "description": (
                    "Exclusive cursor for pagination. Pass the previous response's next_cursor "
                    "to continue with rows whose store_id is greater than this value."
                ),
                "default": 0,
            },
            "roles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional role filter, for example ['user', 'assistant', 'tool', 'system'].",
            },
            "time_from": {
                "type": "number",
                "description": "Optional inclusive minimum message timestamp (Unix seconds).",
            },
            "time_to": {
                "type": "number",
                "description": "Optional inclusive maximum message timestamp (Unix seconds).",
            },
        },
        "required": ["session_id"],
    },
}

LCM_DESCRIBE = {
    "name": "lcm_describe",
    "description": (
        "Inspect a current-session summary node's subtree metadata WITHOUT loading full "
        "content, or inspect an externalized payload ref without opening the "
        "full payload. Returns token counts, child manifest, expand hints, "
        "or externalized payload metadata/preview. Use this to plan retrieval "
        "strategy before spending tokens on lcm_expand inside the active conversation. "
        "For cross-session recall, use session_search first. If called with no "
        "node_id or externalized_ref, returns the top-level DAG overview for "
        "the current session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "integer",
                "description": "Summary node ID to inspect. Omit for session overview.",
            },
            "externalized_ref": {
                "type": "string",
                "description": "Optional externalized payload ref filename to inspect instead of a summary node.",
            },
        },
        "required": [],
    },
}

LCM_EXPAND = {
    "name": "lcm_expand",
    "description": (
        "Recover the original detail behind a summary node, externalized payload, or raw message. "
        "Mode selection (exactly one): node_id (current session only) returns the source messages "
        "or lower-depth summaries that were compacted into a summary node; externalized_ref "
        "(current session only) returns a stored externalized payload's content; store_id returns "
        "a single raw message by store_id and works across sessions, suitable for drilling into "
        "cross-session lcm_grep results. Output is bounded by max_tokens; raw recovery is pageable "
        "via content_offset (and source_offset/source_limit for node_id mode). For Hermes-tracked "
        "session history outside the LCM database, prefer session_search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "integer",
                "description": (
                    "Summary node ID to expand. Current-session only — cross-session DAG expansion "
                    "is not supported in this version."
                ),
            },
            "externalized_ref": {
                "type": "string",
                "description": "Externalized payload ref filename to expand instead of a summary node. Current-session only.",
            },
            "store_id": {
                "type": "integer",
                "description": (
                    "Raw message store_id to fetch. Works across sessions, so a store_id surfaced by "
                    "a cross-session lcm_grep result can be expanded directly. Returns the message's "
                    "content paged by content_offset. If the row references an externalized payload, "
                    "the ref is surfaced via 'externalized_ref'; payload metadata and content are "
                    "session-scoped, so a cross-session row also includes 'externalized_note' "
                    "explaining that the ref is for traceability only and cannot be expanded in this version."
                ),
            },
            "max_tokens": {
                "type": "integer",
                "description": "Token budget for returned content (default 4000)",
                "default": 4000,
            },
            "source_offset": {
                "type": "integer",
                "description": "Zero-based pagination offset into the node's immediate source list (node_id mode only).",
                "default": 0,
            },
            "source_limit": {
                "type": "integer",
                "description": "Maximum number of immediate sources to return from source_offset (node_id mode only). Output still respects max_tokens.",
            },
            "content_offset": {
                "type": "integer",
                "description": "Character offset used to continue an oversized raw message, externalized payload, or store_id-mode message. Use next_content_offset from the previous response.",
                "default": 0,
            },
        },
        "required": [],
    },
}

LCM_STATUS = {
    "name": "lcm_status",
    "description": (
        "Get a quick health overview of the LCM engine for the current session. "
        "Shows compression count, store size, DAG depth distribution, context usage, "
        "active configuration, session/message filter state, and rotate snapshot "
        "state (last_rotate_at, rotate_backup_path, rotate_backup_size when a "
        "/lcm rotate apply has been run). Use this to understand how much history "
        "has been compacted, how the engine is performing, whether the current "
        "session is matched by ignore or stateless session patterns, which message "
        "noise-suppression patterns are loaded, and when the rolling rotate "
        "backup was last written."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

LCM_DOCTOR = {
    "name": "lcm_doctor",
    "description": (
        "Run diagnostics on the LCM database and configuration. Checks database "
        "integrity, detects orphaned DAG nodes, validates configuration, and "
        "reports potential issues. Use this to troubleshoot problems or verify "
        "a healthy setup."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

LCM_REMEMBER = {
    "name": "lcm_remember",
    "description": (
        "Store or update a persistent fact, preference, constraint, or decision that should survive "
        "across sessions. Use this whenever the user states something durable: a preference, "
        "a constraint ('don't push to production without review'), or a decision with context. "
        "Facts are keyed by (scope, key) and upserted. When you update an existing key the response "
        "includes 'previous_value' so you can detect contradictions. "
        "Tag facts with 'tags' to group related items; link related facts with 'related_keys' to "
        "form a lightweight knowledge graph queryable via lcm_recall(related_to=key)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "Stable identifier using dot-notation. "
                    "Examples: 'user.preferred_test_framework', 'project.deadline', "
                    "'constraint.no_production_pushes', 'decision.database_choice'."
                ),
            },
            "value": {
                "type": "string",
                "description": "The fact content. Plain text or JSON string for structured values.",
            },
            "scope": {
                "type": "string",
                "description": (
                    "'global' (default) — visible to all sessions. "
                    "'current' — private to the current session."
                ),
                "default": "global",
            },
            "category": {
                "type": "string",
                "enum": ["fact", "preference", "constraint", "decision"],
                "description": (
                    "'preference': user style/tooling choices. "
                    "'constraint': hard rules the agent must not violate. "
                    "'decision': recorded choices with context. "
                    "'fact': general project or user knowledge (default)."
                ),
                "default": "fact",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of topic tags for grouping related facts. "
                    "E.g. ['auth', 'backend']. Use lcm_recall(tag='auth') to retrieve by tag."
                ),
            },
            "related_keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of fact keys that are semantically related to this one. "
                    "Use lcm_link for bidirectional linking after both facts exist."
                ),
            },
        },
        "required": ["key", "value"],
    },
}

LCM_RECALL = {
    "name": "lcm_recall",
    "description": (
        "Retrieve persistent facts, preferences, constraints, or decisions. "
        "Modes: (1) exact key lookup via 'key'; (2) substring search via 'query'; "
        "(3) filter by 'tag' or 'category'; (4) 'related_to' returns facts linked to a given key; "
        "(5) no filter — returns all facts ordered by most recently updated. "
        "Call at session start to load all durable context before the agent begins working."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Exact key lookup. Returns single fact or found=false.",
            },
            "query": {
                "type": "string",
                "description": "Substring search across keys and values.",
            },
            "related_to": {
                "type": "string",
                "description": (
                    "Return all facts linked to this key via shared tags or related_keys. "
                    "Use to explore the knowledge graph around a topic."
                ),
            },
            "tag": {
                "type": "string",
                "description": "Filter by a specific tag (e.g. 'auth', 'backend').",
            },
            "scope": {
                "type": "string",
                "description": "'global', 'current', or omit to search all scopes.",
            },
            "category": {
                "type": "string",
                "enum": ["fact", "preference", "constraint", "decision"],
                "description": "Optionally filter by category.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum facts to return (default 20, hard cap 100).",
                "default": 20,
            },
        },
        "required": [],
    },
}

LCM_FORGET = {
    "name": "lcm_forget",
    "description": (
        "Delete a stored fact by key and scope. Use when a fact is no longer true — "
        "deadline cancelled, preference changed, constraint lifted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Exact key of the fact to delete."},
            "scope": {
                "type": "string",
                "description": "'global' (default) or 'current' for the current session.",
                "default": "global",
            },
        },
        "required": ["key"],
    },
}

LCM_LINK = {
    "name": "lcm_link",
    "description": (
        "Bidirectionally link two facts via their related_keys. "
        "After linking, lcm_recall(related_to='key1') will return key2 and vice versa. "
        "Use to build explicit causal/dependency connections between facts — e.g. link "
        "'decision.database' to 'constraint.legal_data_residency' to record why that decision was made."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key1": {"type": "string", "description": "First fact key."},
            "key2": {"type": "string", "description": "Second fact key to link to key1."},
            "scope": {
                "type": "string",
                "description": "Scope of both facts. Default 'global'.",
                "default": "global",
            },
        },
        "required": ["key1", "key2"],
    },
}

LCM_SEMANTIC_SEARCH = {
    "name": "lcm_semantic_search",
    "description": (
        "Search DAG summary nodes and facts by semantic similarity using vector embeddings. "
        "Unlike lcm_grep (keyword matching), this finds conceptually related content even when "
        "exact keywords differ — e.g. 'authentication system' finds 'JWT with 24h expiry'. "
        "Requires LCM_EMBEDDING_MODEL to be set (e.g. openai/text-embedding-3-small). "
        "Falls back gracefully with instructions when not configured."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query to find semantically similar content.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default 10, max 50).",
                "default": 10,
            },
            "content_type": {
                "type": "string",
                "enum": ["all", "node", "fact"],
                "description": "'all' searches both DAG nodes and facts; 'node' or 'fact' to restrict.",
                "default": "all",
            },
        },
        "required": ["query"],
    },
}

LCM_EXPAND_QUERY = {
    "name": "lcm_expand_query",
    "description": (
        "Answer a natural-language question using expanded LCM context from the current session. Provide a prompt, and either "
        "query matching summaries to expand or explicit node_ids to inspect. Uses the expansion path "
        "instead of the summarization path so retrieval/synthesis can use a different model or timeout. "
        "Prefer this for questions about the active conversation after compaction; for cross-session recall, use session_search first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The question or task to answer from expanded LCM context",
            },
            "query": {
                "type": "string",
                "description": "Optional search query used to find candidate summaries before expansion",
            },
            "node_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional explicit summary node IDs to expand instead of searching",
            },
            "max_results": {
                "type": "integer",
                "description": "Max candidate summaries to expand when using query (default 5)",
                "default": 5,
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max answer tokens for bounded synthesis returned to the main agent (default 2000)",
                "default": 2000,
            },
            "context_max_tokens": {
                "type": "integer",
                "description": "Expanded serialized summary/raw/child-source/externalized fresh context budget for the auxiliary LLM before it returns the bounded answer (default max(answer max_tokens, 32000 or LCM_EXPANSION_CONTEXT_TOKENS))",
                "default": 32000,
            },
        },
        "required": ["prompt"],
    },
}
