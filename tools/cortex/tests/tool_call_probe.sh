#!/usr/bin/env bash
# Probe a model's tool-calling reliability via Ollama's OpenAI-compat endpoint.
# Usage: tool_call_probe.sh [model] [iterations] [base_url]
# Example: tool_call_probe.sh cortex 20 http://cortex:11434

set -u

MODEL="${1:-cortex-agent}"
N="${2:-10}"
BASE_URL="${3:-http://cortex:11434}"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

PROMPTS=(
  "List the files in /tmp"
  "What's in the directory /var/log?"
  "Show me the contents of /etc/hostname"
  "Read the file /proc/cpuinfo"
  "Check if /usr/bin/python3 exists"
)

REQUEST_TMPL='{
  "model": "%s",
  "stream": false,
  "messages": [{"role": "user", "content": "%s"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "list_dir",
        "description": "List files in a directory",
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"]
        }
      }
    }
  ]
}'

echo "Probing $MODEL with $N requests against $BASE_URL ..."
echo

successes=0
failures=0
total_eval_ms=0

for i in $(seq 1 "$N"); do
  prompt="${PROMPTS[$(( (i - 1) % ${#PROMPTS[@]} ))]}"
  payload=$(printf "$REQUEST_TMPL" "$MODEL" "$prompt")
  start_ns=$(date +%s%N)
  response=$(curl -sS "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data "$payload")
  end_ns=$(date +%s%N)
  elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
  total_eval_ms=$(( total_eval_ms + elapsed_ms ))

  tool_call_name=$(echo "$response" | jq -r '.choices[0].message.tool_calls[0].function.name // empty')
  finish=$(echo "$response" | jq -r '.choices[0].finish_reason // empty')

  if [[ -n "$tool_call_name" ]]; then
    successes=$(( successes + 1 ))
    printf "  [%2d/%d] OK    %-40s -> %s (%dms, finish=%s)\n" "$i" "$N" "$prompt" "$tool_call_name" "$elapsed_ms" "$finish"
  else
    failures=$(( failures + 1 ))
    content=$(echo "$response" | jq -r '.choices[0].message.content // ""' | head -c 80)
    printf "  [%2d/%d] MISS  %-40s -> finish=%s, content=%q\n" "$i" "$N" "$prompt" "$finish" "$content"
  fi
done

echo
pct=$(( successes * 100 / N ))
avg_ms=$(( total_eval_ms / N ))
echo "Result: $successes/$N tool calls (${pct}%), avg ${avg_ms}ms"

if [[ "$successes" -lt "$N" ]]; then
  exit 1
fi
