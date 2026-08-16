{{/*
Expand the name of the chart.
*/}}
{{- define "eggai-multi-agent-chat.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "eggai-multi-agent-chat.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "eggai-multi-agent-chat.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "eggai-multi-agent-chat.labels" -}}
helm.sh/chart: {{ include "eggai-multi-agent-chat.chart" . }}
{{ include "eggai-multi-agent-chat.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "eggai-multi-agent-chat.selectorLabels" -}}
app.kubernetes.io/name: {{ include "eggai-multi-agent-chat.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "eggai-multi-agent-chat.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "eggai-multi-agent-chat.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Per-agent static metadata. Keyed by the values.yaml agent key.

  prefix — the pydantic-settings `env_prefix` on that agent's Settings class.
           MUST match agents/<name>/config.py. An unprefixed env var is
           silently ignored by pydantic, so a mismatch here fails quietly.
  module — argument to `python -m` (the image ENTRYPOINT).
  label  — value of the egg-ai.com/agent pod label. Kept explicit because
           it is a selector value; changing it is a breaking change.
*/}}
{{- define "eggai-multi-agent-chat.agentMeta" -}}
{{- $meta := dict
  "audit"                     (dict "prefix" "AUDIT_"                     "module" "agents.audit.main"                     "label" "audit")
  "billing"                   (dict "prefix" "BILLING_"                   "module" "agents.billing.main"                   "label" "billing")
  "claims"                    (dict "prefix" "CLAIMS_"                    "module" "agents.claims.main"                    "label" "claims")
  "escalation"                (dict "prefix" "ESCALATION_"                "module" "agents.escalation.main"                "label" "escalation")
  "frontend"                  (dict "prefix" "FRONTEND_"                  "module" "agents.frontend.main"                  "label" "frontend")
  "policies"                  (dict "prefix" "POLICIES_"                  "module" "agents.policies.agent.main"            "label" "policies")
  "policiesDocumentIngestion" (dict "prefix" "POLICIES_DOCUMENT_INGESTION_" "module" "agents.policies.ingestion.start_worker" "label" "policies-document-ingestion")
  "triage"                    (dict "prefix" "TRIAGE_"                    "module" "agents.triage.main"                    "label" "triage")
-}}
{{- $found := index $meta . -}}
{{- if not $found -}}
{{- fail (printf "agent %q has no entry in agentMeta; add prefix/module/label in _helpers.tpl" .) -}}
{{- end -}}
{{- $found | toJson -}}
{{- end -}}

{{/*
Env list for one agent, in ascending precedence (later wins in Kubernetes):
  1. globalEnv                   — map, auto-prefixed per agent
  2. extraEnv                    — list, verbatim, supports valueFrom
  3. agents.<name>.environment   — list, verbatim, per-agent override
Usage: include "eggai-multi-agent-chat.agentEnv" (dict "root" $ "agent" $name)
*/}}
{{- define "eggai-multi-agent-chat.agentEnv" -}}
{{- $root := .root -}}
{{- $prefix := (include "eggai-multi-agent-chat.agentMeta" .agent | fromJson).prefix -}}
- name: DSPY_CACHEDIR
  value: /cache/dspy
{{- if $root.Values.monitoring.enabled }}
- name: {{ $prefix }}PROMETHEUS_METRICS_PORT
  value: {{ $root.Values.monitoring.port | quote }}
{{- end }}
{{- range $key, $val := $root.Values.globalEnv }}
{{- if not (empty $val) }}
- name: {{ $prefix }}{{ $key }}
  value: {{ $val | quote }}
{{- end }}
{{- end }}
{{- with $root.Values.extraEnv }}
{{- toYaml . | nindent 0 }}
{{- end }}
{{- with (index $root.Values.agents .agent).environment }}
{{- toYaml . | nindent 0 }}
{{- end }}
{{- end -}}
