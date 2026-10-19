#!/usr/bin/env bash
# Usage: source CWAH/experiment_runner/env.sh ds|jd
trace_load_api_env() {
  local provider="${1:-ds}" runner_dir env_file role base key model protocol
  runner_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  case "${provider}" in
    ds) env_file="${CWAH_DS_ENV_FILE:-${runner_dir}/.env.ds}" ;;
    jd) env_file="${CWAH_JD_ENV_FILE:-${runner_dir}/.env.jd}" ;;
    *) echo 'Usage: source CWAH/experiment_runner/env.sh ds|jd' >&2; return 1 ;;
  esac
  if [[ ! -f "${env_file}" ]]; then
    echo "Missing API configuration: ${env_file}" >&2; return 1
  fi
  local restore_allexport=0
  [[ $- == *a* ]] && restore_allexport=1
  set -a
  source "${env_file}"
  [[ ${restore_allexport} == 1 ]] || set +a
  if [[ "${provider}" == ds ]]; then
    base="${DS_API_BASE:-}"; key="${DS_API_KEY:-}"; model="${DS_MODEL:-}"
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
  if [[ "${provider}" == ds && "${DS_USE_LOCAL_PROXY:-0}" == 1 ]]; then
    export HTTP_PROXY="${DS_HTTP_PROXY:-http://127.0.0.1:7890}"
    export HTTPS_PROXY="${DS_HTTPS_PROXY:-${HTTP_PROXY}}"
    export http_proxy="${HTTP_PROXY}" https_proxy="${HTTPS_PROXY}"
    export CWAH_LOADER_PROXY=1
  fi
  for role in CWAH_AGENT CWAH_HUMAN; do
    export "${role}_API_BASE=${base}" "${role}_API_KEY=${key}" \
           "${role}_LM_ID=${model}" "${role}_WIRE_API=${protocol}"
    if [[ "${provider}" == jd ]]; then
      export "${role}_REASONING_EFFORT=${JD_REASONING_EFFORT:-high}"
      export "${role}_REQUEST_MAX_RETRIES=${JD_REQUEST_MAX_RETRIES:-20}"
      export "${role}_RESPONSES_OUTPUT_BUDGET=${JD_RESPONSES_OUTPUT_BUDGET:-8192}"
    else
      unset "${role}_REASONING_EFFORT" "${role}_REQUEST_MAX_RETRIES" "${role}_RESPONSES_OUTPUT_BUDGET"
    fi
  done
  export CWAH_API_PROVIDER="${provider}"
}
trace_load_api_env "${1:-ds}"
