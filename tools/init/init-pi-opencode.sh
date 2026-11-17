mkdir -p "$HOME/.pi/agent"

# 128K context. Bump to 262144 for full 256K Qwen native — borderline VRAM
# fit on a 32GB card (q4_0 KV cache required either way; see the systemd
# override in dev-init-ai for OLLAMA_KV_CACHE_TYPE).
cat > "$HOME/.pi/agent/models.json" << EOF
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
          "contextWindow": 131072,
          "maxTokens": 32768,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        },
        {
          "id": "qwen-agent",
          "name": "qwen-agent",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 32768,
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
# 128K context. Bump to 262144 for full 256K Qwen native — borderline VRAM
# fit on a 32GB card (q4_0 KV cache required either way; see the systemd
# override in dev-init-ai for OLLAMA_KV_CACHE_TYPE).
# NOTE: opencode's per-model "limit" field is the conventional pattern for
# openai-compatible custom providers; verify with `opencode models` if the
# tool reports a different ceiling than 128K.
cat > "$HOME/.config/opencode/opencode.json" << EOF
{
  "\$schema": "https://opencode.ai/config.json",
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
          "name": "cortex-agent",
          "limit": { "context": 131072, "output": 32768 }
        },
        "qwen-agent": {
          "name": "qwen-agent",
          "limit": { "context": 131072, "output": 32768 }
        }
      }
    }
  }
}
EOF
