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
          "id": "gemma4-agent",
          "name": "gemma4-agent",
          "reasoning": true,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
EOF
# Set gemma4 as pi's default (always loaded); dev can switch to cortex when it's loaded.
PI_SETTINGS="$HOME/.pi/agent/settings.json"
if [[ -f "$PI_SETTINGS" ]]; then
  tmp=$(mktemp)
  python3 -c "
import json, sys
s = json.load(open('$PI_SETTINGS'))
s['defaultProvider'] = 'cortex'
s['defaultModel'] = 'cortex-agent'
json.dump(s, open('$tmp', 'w'), indent=2)
"
  mv "$tmp" "$PI_SETTINGS"
else
  echo '{"defaultProvider":"cortex","defaultModel":"cortex-agent"}' > "$PI_SETTINGS"
fi

# Configure opencode the same way — direct to Ollama, both models declared.
mkdir -p "$HOME/.config/opencode"
cat > "$HOME/.config/opencode/opencode.json" << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
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
        "gemma4-agent": {
          "name": "gemma4-agent"
        }
      }
    }
  }
}
EOF
