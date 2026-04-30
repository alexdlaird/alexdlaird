mkdir -p "$HOME/.pi/agent"
cat > "$HOME/.pi/agent/models.json" << 'EOF'
{
  "providers": {
    "cortex": {
      "baseUrl": "http://cortex:11434/v1",
      "api": "openai-completions",
      "apiKey": "ollama",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "cortex-agent",
          "name": "cortex-agent",
          "reasoning": true,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        },
        {
          "id": "qwen-agent",
          "name": "qwen-agent",
          "reasoning": true,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
EOF
PI_SETTINGS="$HOME/.pi/agent/settings.json"
echo '{"defaultProvider":"cortex","defaultModel":"cortex-agent"}' > "$PI_SETTINGS"

mkdir -p "$HOME/.config/opencode"
cat > "$HOME/.config/opencode/opencode.json" << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "model": "cortex/cortex-agent",
  "provider": {
    "cortex": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Cortex",
      "options": {
        "baseURL": "http://cortex:11434/v1"
      },
      "models": {
        "cortex-agent": {
          "name": "cortex-agent"
        },
        "qwen-agent": {
          "name": "qwen-agent"
        }
      }
    }
  }
}
EOF
