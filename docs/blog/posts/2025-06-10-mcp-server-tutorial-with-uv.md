---
title: "Building a Custom FeatureStoreLite MCP Server Using uv"
date:
  created: 2025-06-10
  updated: 2025-06-10
tags: [guide, mcp]
description: A step-by-step guide that shows how to create your own lightweight feature store MCP server from scratch using FastMCP, run it through **uv**, and integrate it with Claude Desktop. This is a practical example of building a useful MCP server that ML engineers can actually use.
author: Viacheslav Dubrov
---

# Building a Custom FeatureStoreLite MCP Server Using uv

_A step-by-step guide that shows how to create your own lightweight feature store MCP server from scratch using FastMCP, run it through **uv**, and integrate it with Claude Desktop. This is a practical example of building a useful MCP server that ML engineers can actually use._

<!-- more -->

---

## 1. Why build a custom "FeatureStoreLite" MCP server?

Let's create a practical MCP server example that solves a real problem: **feature storage and retrieval for ML pipelines**. Our custom _FeatureStoreLite_ server will be a microservice responsible for storing and retrieving precomputed feature vectors via keys, allowing ML pipelines to share features efficiently without recomputation.

This tutorial demonstrates how to build an **MCP server** that could be useful in a real-world ML pipeline.

## 2. Setup and Installation

First, install **uv** (if you haven't already):

```bash
# macOS/Linux with Homebrew
brew install uv

# Or install directly from the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create your project with virtual environment and dependencies:

```bash
# Create project directory
mkdir mcp-featurestore && cd mcp-featurestore

# Initialize Python project (this creates pyproject.toml)
uv init

# Add dependencies
uv add "mcp[cli]"
```

---

## 3. Implementing our custom FeatureStoreLite server with `FastMCP`

Let's build our MCP server from scratch.

### 3.1. Create the database module

This module handles all database operations for storing and retrieving feature vectors.

Create `database.py` file:

```bash
touch database.py
```

Add the following code to the file:

```python
# database.py

import sqlite3
import os


def get_db_path():
    """Get the database path - always in the script's directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "features.db")


def init_db():
    """Initialize the feature store database"""
    conn = sqlite3.connect(get_db_path())
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features (
            key TEXT PRIMARY KEY,
            vector TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add example features for experimentation
    example_features = [
        (
            "user_123",
            "[0.1, 0.2, -0.5, 0.8, 0.3, -0.1, 0.9, -0.4]",
            '{"type": "user_embedding", "user_id": 123, "age": 25, '
            '"category": "premium"}'
        ),
        (
            "product_abc",
            "[0.7, -0.3, 0.4, 0.1, -0.8, 0.6, 0.2, -0.5]",
            '{"type": "product_embedding", "product_id": "abc", '
            '"price": 29.99, "category": "electronics"}'
        ),
        (
            "doc_guide_001",
            "[-0.2, 0.5, 0.9, -0.1, 0.4, 0.7, -0.6, 0.3]",
            '{"type": "document_embedding", "doc_id": "guide_001", '
            '"title": "Getting Started Guide", "section": "introduction"}'
        ),
        (
            "recommendation_engine",
            "[0.4, 0.8, -0.2, 0.6, -0.7, 0.1, 0.5, -0.9]",
            '{"type": "model_embedding", "model": "collaborative_filter", '
            '"version": "1.2", "accuracy": 0.85}'
        )
    ]

    # Insert example features only if they don't exist
    for key, vector, metadata in example_features:
        existing = conn.execute(
            "SELECT 1 FROM features WHERE key = ?", (key,)
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO features (key, vector, metadata) "
                "VALUES (?, ?, ?)",
                (key, vector, metadata)
            )

    conn.commit()
    conn.close()


def get_db_connection():
    """Get a database connection"""
    return sqlite3.connect(get_db_path())


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!")
```

Run the database initialization:

```bash
uv run python database.py
```

### 3.2. Create the MCP server

Create **`featurestore_server.py`**:

```bash
touch featurestore_server.py
```

Add the following code to the file:

```python
# featurestore_server.py

import json
from mcp.server.fastmcp import FastMCP
from database import get_db_connection, init_db

mcp = FastMCP("FeatureStoreLite")

# Initialize database
init_db()


@mcp.resource("schema://main")
def get_schema() -> str:
    """Provide the database schema as a resource"""
    conn = get_db_connection()
    try:
        schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table'"
        ).fetchall()
        if not schema:
            return "No tables found in database"
        return "\n".join(sql[0] for sql in schema if sql[0])
    except Exception as e:
        return f"Error getting schema: {str(e)}"
    finally:
        conn.close()


@mcp.tool()
def store_feature(key: str, vector: str, metadata: str | None = None) -> str:
    """Store a feature vector with optional metadata"""
    conn = get_db_connection()
    try:
        # Validate vector format (JSON array)
        json.loads(vector)
        conn.execute(
            "INSERT OR REPLACE INTO features (key, vector, metadata) "
            "VALUES (?, ?, ?)",
            (key, vector, metadata)
        )
        conn.commit()
        return f"Feature '{key}' stored successfully"
    except json.JSONDecodeError:
        return "Error: vector must be valid JSON"
    except Exception as e:
        return f"Error storing feature: {str(e)}"
    finally:
        conn.close()


@mcp.tool()
def get_feature(key: str) -> str:
    """Retrieve a feature vector by key"""
    conn = get_db_connection()
    try:
        result = conn.execute(
            "SELECT vector, metadata FROM features WHERE key = ?", (key,)
        ).fetchone()
        if result:
            return json.dumps({
                "key": key,
                "vector": json.loads(result[0]),
                "metadata": json.loads(result[1]) if result[1] else None
            })
        else:
            return f"Feature '{key}' not found"
    except Exception as e:
        return f"Error retrieving feature: {str(e)}"
    finally:
        conn.close()


@mcp.tool()
def list_features() -> str:
    """List all available feature keys"""
    conn = get_db_connection()
    try:
        result = conn.execute(
            "SELECT key, created_at FROM features ORDER BY created_at DESC"
        ).fetchall()
        features = [{"key": row[0], "created_at": row[1]} for row in result]
        return json.dumps(features)
    except Exception as e:
        return f"Error listing features: {str(e)}"
    finally:
        conn.close()


@mcp.resource("features://{key}")
def feature_resource(key: str) -> str:
    """Expose feature data via URI"""
    return get_feature(key)


if __name__ == "__main__":
    mcp.run()
```

Test the server in development mode:

```bash
uv run mcp dev featurestore_server.py
```

---

## 4. Connecting to Claude Desktop

To use the FeatureStoreLite server with Claude Desktop, you need to update your Claude configuration file.

### 4.1. Configuration file location

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

### 4.2. Configuration setup

Add the following configuration to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "featurestore": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp[cli]",
        "mcp",
        "run",
        "/path/to/your/featurestore_server.py"
      ]
    }
  }
}
```

> **💡 Important Tip:** If you are getting errors when connecting to the server, you can use the next command:
>
> ```bash
> uv run mcp install featurestore_server.py
> ```
>
> This command will automatically install and configure the server for Claude Desktop. After running this command, check your Claude Desktop config file to see how the server has been configured.
>
> This is often the easiest way to get started, especially if you're having trouble with manual configuration!

### 4.3 Testing the server

After updating the config:

1. **Restart Claude Desktop** completely (quit and reopen)
2. Look for connection status in Claude's interface
3. Try asking Claude to interact with your feature store

Example queries to test:

- "Show me the database schema for the feature store"

![Question 1](https://slavadubrov.github.io/blog/assets/2025-06-10-mcp-server-tutorial-with-uv/question-1.png){: style="width:600px;max-width:100%;height:auto;"}

- "List all available features in the store"

![Question 2](https://slavadubrov.github.io/blog/assets/2025-06-10-mcp-server-tutorial-with-uv/question-2.png){: style="width:600px;max-width:100%;height:auto;"}

- "Retrieve the feature vector for product_abc"

![Question 3](https://slavadubrov.github.io/blog/assets/2025-06-10-mcp-server-tutorial-with-uv/question-3.png){: style="width:600px;max-width:100%;height:auto;"}

- Try your own queries!

### 4.4. Production deployment considerations

Please note that this is a simple example to demonstrate the MCP server usage and not production-ready.

For production use, consider:

- Using a proper database (PostgreSQL, MySQL) instead of SQLite
- Adding authentication and authorization
- Implementing proper logging and monitoring
- Adding data validation and sanitization
- Using environment variables for configuration

---

## 5. Alternative client configurations

### 5.1. Generic MCP client configuration

For other MCP clients, you can use exactly the same configuration pattern as we did for Claude Desktop:

```json
{
  "mcpServers": {
    "FeatureStoreLite": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp[cli]",
        "mcp",
        "run",
        "path/to/featurestore_server.py"
      ]
    }
  }
}
```

---

## 6. Observability and debugging with Inspector

The MCP development tools provide excellent observability features.

### 6.1. Running in development mode

```bash
uv run mcp dev featurestore_server.py
```

This starts the server with the MCP Inspector, which provides:

- Real-time request/response monitoring
- Tool and resource exploration
- Interactive testing interface
- Performance metrics

### 6.2. Using the Inspector

When running in development mode, the Inspector is available at the URL shown in the terminal output (typically `http://localhost:6274`).

The Inspector allows you to:

- **Browse Resources**: View available resources like the database schema
- **Test Tools**: Interactively test each tool with different parameters
- **Monitor Traffic**: See all MCP protocol messages in real-time
- **Debug Issues**: Identify problems with tool calls or resource access

![Inspector](https://slavadubrov.github.io/blog/assets/2025-06-10-mcp-server-tutorial-with-uv/inspector.png){: style="width:600px;max-width:100%;height:auto;"}

You can check manually the resources available in the server:

![Resources](https://slavadubrov.github.io/blog/assets/2025-06-10-mcp-server-tutorial-with-uv/inspector-resources.png){: style="width:600px;max-width:100%;height:auto;"}

As well as the tools available:

![Tools](https://slavadubrov.github.io/blog/assets/2025-06-10-mcp-server-tutorial-with-uv/inspector-tools.png){: style="width:600px;max-width:100%;height:auto;"}

---

## 7. Conclusion

In this tutorial, we built a custom FeatureStoreLite MCP server using FastMCP, ran it through uv, and integrated it with Claude Desktop. We also explored how to use the `mcp inspector` to see the server's capabilities and the requests and responses it is sending and receiving.

## 8. References

- [The repo of this tutorial example](https://github.com/slavadubrov/mcp-featurestore)
- [Introduction to MCP](https://www.anthropic.com/news/model-context-protocol)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Claude Desktop](https://claude.ai/download)
- [uv](https://docs.astral.sh/uv/)
