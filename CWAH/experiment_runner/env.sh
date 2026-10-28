#!/usr/bin/env bash
# Usage: source CWAH/experiment_runner/env.sh self|ds|jd|qwen
trace_load_api_env() {
  local provider="${1:-self}" runner_dir env_file role base key model protocol
  runner_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  case "${provider}" in
    self) env_file="${CWAH_SELF_ENV_FILE:-${runner_dir}/.env.self}" ;;
    ds) env_file="${CWAH_DS_ENV_FILE:-${runner_dir}/.env.ds}" ;;
    jd) env_file="${CWAH_JD_ENV_FILE:-${runner_dir}/.env.jd}" ;;
    qwen) env_file="${CWAH_QWEN_ENV_FILE:-${runner_dir}/.env.qwen}" ;;
    *) echo 'Usage: source CWAH/experiment_runner/env.sh self|ds|jd|qwen' >&2; return 1 ;;
  esac
  if [[ ! -f "${env_file}" ]]; then
    echo "Missing API configuration: ${env_file}" >&2; return 1
  fi
  local restore_allexport=0
  [[ $- == *a* ]] && restore_allexport=1
  set -a
  source "${env_file}"
  [[ ${restore_allexport} == 1 ]] || set +a
  if [[ "${provider}" == ds || "${provider}" == self ]]; then
    base="${DS_API_BASE:-}"; key="${DS_API_KEY:-}"; model="${DS_MODEL:-}"
    protocol=chat
  elif [[ "${provider}" == qwen ]]; then
    base="${QWEN_API_BASE:-}"; key="${QWEN_API_KEY:-}"; model="${QWEN_MODEL:-}"
    protocol=chat
  else
    base="${JD_API_BASE:-}"; key="${PROXY_OPENAI_API_KEY:-}"; model="${JD_MODEL:-}"
    protocol=responses
  fi
  if [[ -z "${base}" || -z "${model}" || -z "${key}" ]]; then
    echo "Fill the API base, model and key in ${env_file} first." >&2; return 1
  fi
  if [[ "${CWAH_LOADER_PROXY:-0}" == 1 ]]; then
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy CWAH_LOADER_PROXY
  fi
  if [[ ( "${provider}" == ds || "${provider}" == self ) && "${DS_USE_LOCAL_PROXY:-0}" == 1 ]]; then
    export HTTP_PROXY="${DS_HTTP_PROXY:-http://127.0.0.1:7890}"
    export HTTPS_PROXY="${DS_HTTPS_PROXY:-${HTTP_PROXY}}"
    export http_proxy="${HTTP_PROXY}" https_proxy="${HTTPS_PROXY}"
    export CWAH_LOADER_PROXY=1
  fi
  if [[ "${provider}" == qwen ]]; then
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
    export no_proxy="${NO_PROXY}"
  fi
  for role in CWAH_AGENT CWAH_HUMAN; do
    export "${role}_API_BASE=${base}" "${role}_API_KEY=${key}" \
           "${role}_LM_ID=${model}" "${role}_WIRE_API=${protocol}"
    if [[ "${provider}" == qwen ]]; then
      export "${role}_CHAT_ENABLE_THINKING=${QWEN_ENABLE_THINKING:-0}"
      if [[ "${role}" == CWAH_AGENT ]]; then
        export "${role}_CHAT_THINKING_ACTIVE=${QWEN_ALICE_ACTIVE_THINKING:-${QWEN_ENABLE_THINKING:-0}}"
        export "${role}_CHAT_THINKING_UPDATER=${QWEN_ALICE_UPDATER_THINKING:-${QWEN_ENABLE_THINKING:-0}}"
        export "${role}_CHAT_THINKING_REACTIVE=${QWEN_ALICE_REACTIVE_THINKING:-${QWEN_ENABLE_THINKING:-0}}"
      else
        export "${role}_CHAT_THINKING_ACTIVE=${QWEN_BOB_ACTIVE_THINKING:-${QWEN_ENABLE_THINKING:-0}}"
        export "${role}_CHAT_THINKING_UPDATER=0"
        export "${role}_CHAT_THINKING_REACTIVE=${QWEN_BOB_REACTIVE_THINKING:-${QWEN_ENABLE_THINKING:-0}}"
      fi
      export "${role}_CHAT_REASONING_EFFORT=${QWEN_REASONING_EFFORT:-low}"
      export "${role}_CHAT_THINKING_TOKEN_BUDGET=${QWEN_THINKING_TOKEN_BUDGET:-512}"
      export "${role}_CHAT_UPDATER_THINKING_TOKEN_BUDGET=${QWEN_UPDATER_THINKING_TOKEN_BUDGET:-256}"
      export "${role}_CHAT_UPDATER_MAX_TOKENS=${QWEN_UPDATER_MAX_TOKENS:-2048}"
      export "${role}_CHAT_REQUEST_TIMEOUT=${QWEN_CHAT_REQUEST_TIMEOUT:-120}"
      export "${role}_REQUEST_MAX_RETRIES=${QWEN_REQUEST_MAX_RETRIES:-2}"
    else
      unset "${role}_CHAT_ENABLE_THINKING" "${role}_CHAT_REASONING_EFFORT" \
            "${role}_CHAT_THINKING_ACTIVE" "${role}_CHAT_THINKING_UPDATER" \
            "${role}_CHAT_THINKING_REACTIVE" \
            "${role}_CHAT_THINKING_TOKEN_BUDGET" "${role}_CHAT_UPDATER_THINKING_TOKEN_BUDGET" \
            "${role}_CHAT_UPDATER_MAX_TOKENS" "${role}_CHAT_REQUEST_TIMEOUT"
    fi
    if [[ "${provider}" == jd ]]; then
      export "${role}_REASONING_EFFORT=${JD_REASONING_EFFORT:-high}"
      export "${role}_REQUEST_MAX_RETRIES=${JD_REQUEST_MAX_RETRIES:-20}"
      export "${role}_RESPONSES_OUTPUT_BUDGET=${JD_RESPONSES_OUTPUT_BUDGET:-8192}"
    elif [[ "${provider}" == qwen ]]; then
      unset "${role}_REASONING_EFFORT" "${role}_RESPONSES_OUTPUT_BUDGET"
    else
      unset "${role}_REASONING_EFFORT" "${role}_REQUEST_MAX_RETRIES" "${role}_RESPONSES_OUTPUT_BUDGET"
    fi
  done
  export CWAH_API_PROVIDER="${provider}"
}
trace_load_api_env "${1:-self}"
