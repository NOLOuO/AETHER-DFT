from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote
from wsgiref.simple_server import make_server
from wsgiref.util import setup_testing_defaults

from jinja2 import DictLoader, Environment, select_autoescape

from dft_app.storage import RecordStore
from dft_app.web.background_jobs import create_job, list_jobs, load_job, spawn_job_worker, write_job
from dft_app.web.services import load_run_detail_view


TEMPLATES = {
    "base.html": """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }} | semi_auto_dft</title>
  <style>
    *,*::before,*::after{box-sizing:border-box}
    :root{
      --bg:#f0f4f8; --surface:#fff; --surface-alt:#f8fafc;
      --text:#1e293b; --text-secondary:#64748b; --text-muted:#94a3b8;
      --border:#e2e8f0; --border-light:#f1f5f9;
      --primary:#2563eb; --primary-hover:#1d4ed8; --primary-light:#dbeafe; --primary-text:#1e40af;
      --success:#059669; --success-light:#d1fae5; --success-text:#065f46;
      --warning:#d97706; --warning-light:#fef3c7; --warning-text:#92400e;
      --danger:#dc2626; --danger-light:#fee2e2; --danger-text:#991b1b;
      --info:#0284c7; --info-light:#e0f2fe; --info-text:#075985;
      --nav:#0f172a; --nav-text:#e2e8f0;
      --radius:10px; --radius-lg:14px;
      --shadow:0 1px 3px rgba(15,23,42,.06),0 1px 2px rgba(15,23,42,.04);
      --shadow-md:0 4px 12px rgba(15,23,42,.08);
      --transition:150ms ease;
    }
    body{font-family:"Microsoft YaHei","PingFang SC","Helvetica Neue",sans-serif;margin:0;background:var(--bg);color:var(--text);line-height:1.6}

    /* --- Header & Nav --- */
    header{background:var(--nav);color:#fff;padding:0 24px;display:flex;align-items:center;justify-content:space-between;height:56px;position:sticky;top:0;z-index:100}
    header h1{margin:0;font-size:16px;font-weight:600;letter-spacing:.5px;white-space:nowrap}
    header h1 span{color:var(--primary);margin-right:2px}
    nav{display:flex;gap:4px}
    nav a{color:var(--nav-text);text-decoration:none;padding:6px 14px;border-radius:6px;font-size:14px;transition:background var(--transition),color var(--transition)}
    nav a:hover,nav a.active{background:rgba(255,255,255,.12);color:#fff}

    /* --- Layout --- */
    main{padding:24px;max-width:1280px;margin:0 auto}
    .page-title{font-size:22px;font-weight:700;margin:0 0 20px;display:flex;align-items:center;gap:10px}
    .page-title .back{color:var(--text-secondary);text-decoration:none;font-size:18px;transition:color var(--transition)}
    .page-title .back:hover{color:var(--primary)}

    /* --- Grid --- */
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:20px}
    .grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
    .split{display:grid;grid-template-columns:1fr 1fr;gap:20px}
    @media(max-width:900px){.split,.grid,.grid-3{grid-template-columns:1fr}}

    /* --- Card --- */
    .card{background:var(--surface);border-radius:var(--radius-lg);padding:22px 24px;box-shadow:var(--shadow);border:1px solid var(--border-light)}
    .card h2{margin:0 0 16px;font-size:17px;font-weight:700;display:flex;align-items:center;gap:8px}
    .card h2 .icon{font-size:20px}
    .card h3{margin:16px 0 10px;font-size:15px;font-weight:600;color:var(--text-secondary)}

    /* --- Notice / Alert --- */
    .notice{background:var(--info-light);color:var(--info-text);padding:14px 18px;border-radius:var(--radius);margin-bottom:20px;font-size:14px;line-height:1.7;border-left:4px solid var(--info)}
    .notice-success{background:var(--success-light);color:var(--success-text);border-left-color:var(--success)}
    .notice-warning{background:var(--warning-light);color:var(--warning-text);border-left-color:var(--warning)}

    /* --- Form --- */
    label{display:block;font-weight:600;margin-top:14px;font-size:14px;color:var(--text)}
    label .hint{font-weight:400;color:var(--text-muted);font-size:12px;margin-left:6px}
    input[type="text"],input[type="number"],textarea,select{
      width:100%;margin-top:6px;padding:10px 14px;border:1px solid var(--border);border-radius:var(--radius);
      background:var(--surface);font-size:14px;color:var(--text);transition:border-color var(--transition),box-shadow var(--transition);
      font-family:inherit}
    input:focus,textarea:focus,select:focus{outline:none;border-color:var(--primary);box-shadow:0 0 0 3px var(--primary-light)}
    textarea{min-height:100px;resize:vertical}
    .checkbox-row{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px}
    .checkbox-row label{font-weight:500;display:flex;align-items:center;gap:6px;margin-top:0;font-size:14px;cursor:pointer}
    .checkbox-row input[type="checkbox"]{width:16px;height:16px;accent-color:var(--primary)}

    /* --- Buttons --- */
    button,.btn{
      display:inline-flex;align-items:center;gap:6px;
      margin-top:0;padding:9px 16px;border-radius:var(--radius);border:none;
      font-size:13px;font-weight:600;cursor:pointer;text-decoration:none;
      transition:background var(--transition),transform var(--transition),box-shadow var(--transition);
      font-family:inherit}
    button:active,.btn:active{transform:scale(.97)}
    .btn-primary,button[type="submit"]{background:var(--primary);color:#fff}
    .btn-primary:hover,button[type="submit"]:hover{background:var(--primary-hover);box-shadow:var(--shadow-md)}
    .btn-secondary,button.secondary{background:var(--surface);color:var(--text);border:1px solid var(--border)}
    .btn-secondary:hover,button.secondary:hover{background:var(--surface-alt);border-color:var(--text-muted)}
    .btn-success{background:var(--success);color:#fff}
    .btn-success:hover{background:#047857}
    .btn-warning,button.warn{background:var(--warning);color:#fff}
    .btn-warning:hover,button.warn:hover{background:#b45309}
    .btn-danger{background:var(--danger);color:#fff}
    .btn-danger:hover{background:#b91c1c}
    .btn-sm{padding:6px 12px;font-size:12px}

    /* --- Actions bar --- */
    .actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
    .actions form{margin:0}
    .actions button{margin-top:0}

    /* --- Table --- */
    table{width:100%;border-collapse:collapse;font-size:14px}
    thead th{text-align:left;padding:10px 12px;background:var(--surface-alt);color:var(--text-secondary);font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:.3px;border-bottom:2px solid var(--border)}
    tbody td{padding:10px 12px;border-bottom:1px solid var(--border-light);vertical-align:middle}
    tbody tr:hover{background:var(--surface-alt)}
    td a{color:var(--primary);text-decoration:none;font-weight:600}
    td a:hover{text-decoration:underline}
    .empty-state{text-align:center;padding:40px 20px;color:var(--text-muted)}

    /* --- Tags / Badges --- */
    .tag{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;letter-spacing:.2px}
    .tag{background:var(--primary-light);color:var(--primary-text)}
    .tag.ready,.tag.completed,.tag.analyzed{background:var(--success-light);color:var(--success-text)}
    .tag.running,.tag.submitted,.tag.monitoring{background:var(--warning-light);color:var(--warning-text)}
    .tag.failed,.tag.blocked,.tag.error{background:var(--danger-light);color:var(--danger-text)}
    .tag.pending,.tag.created{background:#f1f5f9;color:#475569}
    .tag::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:currentColor;opacity:.6}

    /* --- Key-Value display --- */
    .kv{display:grid;grid-template-columns:auto 1fr;gap:6px 16px;font-size:14px;align-items:baseline}
    .kv dt{font-weight:600;color:var(--text-secondary);white-space:nowrap}
    .kv dd{margin:0;word-break:break-all}

    /* --- Pre / Code --- */
    pre{white-space:pre-wrap;word-break:break-word;background:var(--nav);color:#e2e8f0;padding:16px 18px;border-radius:var(--radius);overflow:auto;font-size:13px;line-height:1.6;margin:12px 0 0}
    code{font-family:Consolas,"Source Code Pro",monospace;font-size:13px}
    .inline-code{background:var(--surface-alt);color:var(--primary-text);padding:2px 8px;border-radius:4px;font-size:13px;border:1px solid var(--border)}

    /* --- Workflow subtask cards --- */
    .subtask-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px}
    .subtask-card h4{margin:0 0 8px;font-size:14px;font-weight:600;display:flex;align-items:center;justify-content:space-between}
    .subtask-card .desc{font-size:13px;color:var(--text-secondary);margin:0}

    /* --- Separator --- */
    hr{border:none;border-top:1px solid var(--border);margin:20px 0}

    /* --- Collapsible --- */
    details{margin-top:12px}
    details summary{cursor:pointer;font-weight:600;font-size:14px;color:var(--text-secondary);padding:8px 0;user-select:none}
    details summary:hover{color:var(--primary)}

    /* --- Footer --- */
    footer{text-align:center;padding:24px;color:var(--text-muted);font-size:12px}
  </style>
</head>
<body>
  <header>
    <h1><span>DFT</span> semi_auto_dft</h1>
    <nav>
      <a href="/" {% if active_nav == 'home' %}class="active"{% endif %}>首页</a>
      <a href="/runs" {% if active_nav == 'runs' %}class="active"{% endif %}>Runs 列表</a>
    </nav>
  </header>
  <main>
    {% block content %}{% endblock %}
  </main>
  <footer>semi_auto_dft Web UI &middot; 本地轻量可视化界面</footer>
</body>
</html>
""",
    "home.html": """
{% extends "base.html" %}
{% block content %}
<h1 class="page-title">DFT 任务控制台</h1>

<div class="notice">当前界面是对 CLI / RecordStore 的可视化封装：提交任务后，围绕 run_root 执行 submit / monitor / fetch / parse / analyze / adsorption-workflow / dft_tools explain，并逐步完成 adsorption 全主线。</div>

<section class="card" style="margin-bottom:16px">
  <h2><span class="icon">◎</span> adsorption 主线步骤概览</h2>
  <ol style="padding-left:20px; line-height:1.8">
    <li>输入 adsorption 任务与 slab 结构，建议先 dry-run。</li>
    <li>生成 candidates 并确认 selected candidate。</li>
    <li>围绕主 run_root 推进 workflow submit / monitor / fetch / parse-analyze。</li>
    <li>完成 dft_tools explain，并继续 knowledge backflow。</li>
  </ol>
  <p class="muted" style="margin:0">如果你不确定下一步做什么，先进入 run 详情页看“推荐下一步”。</p>
</section>

<div class="grid">
  <section class="card">
    <h2><span class="icon">+</span> 新建任务</h2>
    <form method="post" action="/actions/run">
      <label>任务描述</label>
      <textarea name="prompt" placeholder="例如：计算 H2O 在 Cu(111) 上的吸附能"></textarea>

      <label>结构路径 <span class="hint">POSCAR / CIF / XSD</span></label>
      <input type="text" name="structure_path" placeholder="例如：F:\\DFTauto\\...\\slab.vasp">

      <label>材料名 <span class="hint">可选</span></label>
      <input type="text" name="material" placeholder="例如：Cu(111)">

      <div class="split" style="gap:12px">
        <div>
          <label>提交 Profile</label>
          <select name="submit_profile">
            <option value="">不指定</option>
            {% for profile in submit_profiles %}
            <option value="{{ profile }}">{{ profile }}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Candidate ID <span class="hint">吸附主线</span></label>
          <input type="text" name="selected_candidate_id" placeholder="ontop_01_upright">
        </div>
      </div>

      <div class="checkbox-row">
        <label><input type="checkbox" name="dry_run" value="1"> dry-run</label>
        <label><input type="checkbox" name="submit" value="1"> 直接提交</label>
        <label><input type="checkbox" name="remote" value="1"> 远程模式</label>
      </div>
      <div style="margin-top:16px">
        <button type="submit">执行 dft run</button>
      </div>
    </form>
  </section>

  <section class="card">
    <h2><span class="icon">~</span> 最近 Runs</h2>
    {% if recent_runs %}
      <table>
        <thead><tr><th>Run</th><th>Task</th><th>状态</th><th>更新时间</th></tr></thead>
        <tbody>
        {% for run in recent_runs %}
          <tr>
            <td><a href="/run-detail?run_root={{ run.run_root | urlencode }}">{{ run.run_id[:12] }}</a></td>
            <td><span class="inline-code">{{ run.task_id[:16] }}</span></td>
            <td><span class="tag {{ run.overall_status }}">{{ run.overall_status }}</span></td>
            <td style="color:var(--text-muted);font-size:13px">{{ run.updated_at }}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    {% else %}
      <div class="empty-state">
        <p>当前没有 runs</p>
        <p style="font-size:13px">在左侧表单提交任务后，run 记录会显示在这里。</p>
      </div>
    {% endif %}
  </section>
</div>
{% endblock %}
""",
    "runs.html": """
{% extends "base.html" %}
{% block content %}
<h1 class="page-title">Runs 列表</h1>
<section class="card">
  {% if runs %}
  <table>
    <thead><tr><th>Run ID</th><th>Task ID</th><th>状态</th><th>阶段</th><th>Job ID</th><th>更新时间</th></tr></thead>
    <tbody>
    {% for run in runs %}
      <tr>
        <td><a href="/run-detail?run_root={{ run.run_root | urlencode }}">{{ run.run_id }}</a></td>
        <td><span class="inline-code">{{ run.task_id }}</span></td>
        <td><span class="tag {{ run.overall_status }}">{{ run.overall_status }}</span></td>
        <td>{{ run.current_phase or "-" }}</td>
        <td>{{ run.scheduler_job_id or "-" }}</td>
        <td style="color:var(--text-muted);font-size:13px">{{ run.updated_at }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}
  <div class="empty-state"><p>暂无 runs 记录。</p></div>
  {% endif %}
</section>
{% endblock %}
""",
    "run_detail.html": """
{% extends "base.html" %}
{% block content %}
<h1 class="page-title">
  <a href="/runs" class="back">&larr;</a>
  Run 详情
  <span class="tag {{ run_record.overall_status }}">{{ run_record.overall_status }}</span>
</h1>

<div class="split">
  <section class="card">
    <h2>Run 概览</h2>
    <dl class="kv">
      <dt>Run ID</dt>
      <dd><span class="inline-code">{{ run_record.run_id }}</span></dd>
      <dt>Task ID</dt>
      <dd><span class="inline-code">{{ run_record.task_id }}</span></dd>
      <dt>当前阶段</dt>
      <dd>{{ run_record.current_phase or "-" }}</dd>
      <dt>Job ID</dt>
      <dd>{{ run_record.scheduler_job_id or "-" }}</dd>
      <dt>run_root</dt>
      <dd style="font-size:12px;color:var(--text-muted);word-break:break-all">{{ run_root }}</dd>
    </dl>

    {% if next_step_cards %}
    <hr>
    <h3>推荐下一步</h3>
    <div class="grid" style="grid-template-columns:1fr;gap:10px">
      {% for item in next_step_cards %}
      <div class="notice" style="margin:0">
        <strong>{{ item.title }}</strong><br>
        {{ item.description }}
      </div>
      {% endfor %}
    </div>
    {% endif %}

    <hr>
    <h3>Pipeline 操作</h3>
    <div class="actions">
      <form method="post" action="/actions/step">
        <input type="hidden" name="run_root" value="{{ run_root }}">
        <input type="hidden" name="phase" value="submit">
        <button type="submit" class="btn-success btn-sm">提交</button>
      </form>
      <form method="post" action="/actions/step">
        <input type="hidden" name="run_root" value="{{ run_root }}">
        <input type="hidden" name="phase" value="monitor">
        <button type="submit" class="btn-sm secondary">监控</button>
      </form>
      <form method="post" action="/actions/step">
        <input type="hidden" name="run_root" value="{{ run_root }}">
        <input type="hidden" name="phase" value="parse">
        <button type="submit" class="btn-sm secondary">解析</button>
      </form>
      <form method="post" action="/actions/step">
        <input type="hidden" name="run_root" value="{{ run_root }}">
        <input type="hidden" name="phase" value="analyze">
        <button type="submit" class="btn-sm secondary">分析</button>
      </form>
      <form method="post" action="/actions/fetch">
        <input type="hidden" name="run_root" value="{{ run_root }}">
        <button type="submit" class="btn-sm btn-warning">远程拉回</button>
      </form>
      <form method="post" action="/actions/dft-tools-explain">
        <input type="hidden" name="run_root" value="{{ run_root }}">
        <button type="submit" class="btn-sm secondary">dft_tools 结果解释</button>
      </form>
    </div>
  </section>

  <section class="card">
    <h2>Metadata</h2>
    {% if experiment_spec %}
    <dl class="kv">
      <dt>材料</dt>
      <dd>{{ experiment_spec.material_name or "-" }}</dd>
      <dt>任务类型</dt>
      <dd><span class="inline-code">{{ experiment_spec.task_type or "-" }}</span></dd>
      <dt>泛函</dt>
      <dd>{{ experiment_spec.functional or "-" }}</dd>
      <dt>结构来源</dt>
      <dd>{{ experiment_spec.structure_source or "-" }}</dd>
      <dt>workflow</dt>
      <dd>{{ experiment_spec.workflow | join(" &rarr; ") if experiment_spec.workflow else "-" }}</dd>
    </dl>
    {% endif %}
    <details>
      <summary>完整 Metadata JSON</summary>
      <pre>{{ payload_json }}</pre>
    </details>
  </section>
</div>

<section class="card" style="margin-top:20px">
  <h2><span class="icon">⇢</span> 产品主线视图</h2>
  <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px">
    {% for step in user_flow_steps %}
    <div class="subtask-card">
      <h4>{{ step.name }} <span class="tag {{ step.status }}">{{ step.status }}</span></h4>
      <p class="desc">{{ step.description }}</p>
    </div>
    {% endfor %}
  </div>
</section>

<section class="card" style="margin-top:20px">
  <h2><span class="icon">◇</span> Candidate 阶段</h2>
  <dl class="kv">
    <dt>当前状态</dt>
    <dd><span class="tag {{ candidate_state.status }}">{{ candidate_state.status }}</span></dd>
    <dt>候选数</dt>
    <dd>{{ candidate_state.candidate_count or 0 }}</dd>
    <dt>当前 selected candidate</dt>
    <dd>{{ candidate_state.selected_candidate_id or "-" }}</dd>
    <dt>下一步</dt>
    <dd>{{ candidate_state.next_step }}</dd>
  </dl>
  {% if candidate_state.candidate_ids %}
  <h3>候选 ID 示例</h3>
  <div class="actions">
    {% for cid in candidate_state.candidate_ids %}
      <span class="inline-code">{{ cid }}</span>
    {% endfor %}
  </div>
  {% endif %}
  {% if candidate_state.manifest_json %}
  <p class="muted">manifest.json: {{ candidate_state.manifest_json }}</p>
  {% endif %}
  {% if candidate_state.manifest_md %}
  <p class="muted">manifest.md: {{ candidate_state.manifest_md }}</p>
  {% endif %}
</section>

{% if dft_tools_explain %}
<section class="card" style="margin-top:20px">
  <h2><span class="icon">🧠</span> dft_tools 结果解释</h2>
  <dl class="kv">
    <dt>状态判断</dt>
    <dd>{{ dft_tools_explain.status_judgement or "-" }}</dd>
    <dt>Provider / Model</dt>
    <dd>{{ dft_tools_explain.provider or "none" }} / {{ dft_tools_explain.model or "none" }}</dd>
  </dl>
  {% if dft_tools_explain.likely_causes %}
  <h3>主要原因</h3>
  <ul>
  {% for item in dft_tools_explain.likely_causes %}
    <li>{{ item }}</li>
  {% endfor %}
  </ul>
  {% endif %}
  {% if dft_tools_explain.next_actions %}
  <h3>下一步建议</h3>
  <ul>
  {% for item in dft_tools_explain.next_actions %}
    <li>{{ item }}</li>
  {% endfor %}
  </ul>
  {% endif %}
  {% if explain_state.result_path %}
  <p class="muted">结果入口：{{ explain_state.result_path }}</p>
  {% endif %}
  <details>
    <summary>dft_tools explain 完整 JSON</summary>
    <pre>{{ dft_tools_explain_json }}</pre>
  </details>
</section>
{% endif %}

{% if dft_tools_kb_ingest_result %}
<section class="card" style="margin-top:20px">
  <h2><span class="icon">🗂️</span> dft_tools 知识库回流</h2>
  <dl class="kv">
    <dt>回流状态</dt>
    <dd>{{ dft_tools_kb_ingest_result.status or "-" }}</dd>
    <dt>自动回流</dt>
    <dd>{{ dft_tools_kb_ingest_result.enabled }}</dd>
  </dl>
  {% if dft_tools_kb_ingest_result.error %}
  <div class="notice notice-warning">{{ dft_tools_kb_ingest_result.error }}</div>
  {% endif %}
  {% if backflow_state.result_path %}
  <p class="muted">结果入口：{{ backflow_state.result_path }}</p>
  {% endif %}
  <details>
    <summary>KB ingest 完整 JSON</summary>
    <pre>{{ dft_tools_kb_ingest_json }}</pre>
  </details>
</section>
{% endif %}

{% if workflow_status %}
<section class="card" style="margin-top:20px">
  <h2><span class="icon">&harr;</span> Adsorption Workflow</h2>

  <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
    <span>workflow 状态:</span>
    <span class="tag {{ workflow_status.status }}">{{ workflow_status.status }}</span>
  </div>

  {% if workflow_status.recommended_next_steps %}
  <div class="notice">
    <strong>下一步建议：</strong>
    <ul style="margin:6px 0 0;padding-left:20px">
    {% for item in workflow_status.recommended_next_steps %}
      <li>{{ item }}</li>
    {% endfor %}
    </ul>
  </div>
  {% endif %}

  {% if workflow_status.subtask_statuses %}
  <div class="grid-3" style="margin-bottom:16px">
    {% for name, info in workflow_status.subtask_statuses.items() %}
    <div class="subtask-card">
      <h4>
        {{ name }}
        <span class="tag {{ info.status or 'pending' }}">{{ info.status or "pending" }}</span>
      </h4>
      {% if info.total_energy is not none %}
      <p class="desc">E = {{ "%.6f" | format(info.total_energy) }} eV</p>
      {% else %}
      <p class="desc">能量: 待计算</p>
      {% endif %}
    </div>
    {% endfor %}
  </div>
  {% endif %}

  <div class="actions">
    <form method="post" action="/actions/workflow">
      <input type="hidden" name="run_root" value="{{ run_root }}">
      <input type="hidden" name="action" value="status">
      <button type="submit" class="btn-sm secondary">刷新 workflow 状态</button>
    </form>
    <form method="post" action="/actions/workflow">
      <input type="hidden" name="run_root" value="{{ run_root }}">
      <input type="hidden" name="action" value="submit">
      <button type="submit" class="btn-sm btn-success">workflow 提交</button>
    </form>
    <form method="post" action="/actions/workflow">
      <input type="hidden" name="run_root" value="{{ run_root }}">
      <input type="hidden" name="action" value="monitor">
      <button type="submit" class="btn-sm secondary">workflow 监控</button>
    </form>
    <form method="post" action="/actions/workflow">
      <input type="hidden" name="run_root" value="{{ run_root }}">
      <input type="hidden" name="action" value="fetch">
      <button type="submit" class="btn-sm btn-warning">workflow 拉回</button>
    </form>
    <form method="post" action="/actions/workflow">
      <input type="hidden" name="run_root" value="{{ run_root }}">
      <input type="hidden" name="action" value="parse-analyze">
      <button type="submit" class="btn-sm secondary">workflow 解析/分析</button>
    </form>
  </div>

  <details>
    <summary>Workflow 完整 JSON</summary>
    <pre>{{ workflow_json }}</pre>
  </details>
</section>
{% endif %}
{% endblock %}
""",
    "command_result.html": """
{% extends "base.html" %}
{% block content %}
<h1 class="page-title">
  {% if back_url %}<a href="{{ back_url }}" class="back">&larr;</a>{% endif %}
  {{ title }}
</h1>

<section class="card">
  <dl class="kv">
    <dt>命令</dt>
    <dd><code class="inline-code" style="font-size:12px">{{ command }}</code></dd>
    <dt>返回码</dt>
    <dd>
      {% if returncode == 0 %}
        <span class="tag ready">{{ returncode }}</span>
      {% else %}
        <span class="tag failed">{{ returncode }}</span>
      {% endif %}
    </dd>
  </dl>
</section>

<div class="split" style="margin-top:20px">
  <section class="card">
    <h3 style="margin-top:0">stdout</h3>
    <pre>{{ stdout or "(empty)" }}</pre>
  </section>
  <section class="card">
    <h3 style="margin-top:0">stderr</h3>
    <pre>{{ stderr or "(empty)" }}</pre>
  </section>
</div>

{% if dft_tools_explain %}
<section class="card" style="margin-top:20px">
  <h2><span class="icon">🧠</span> dft_tools 结果解释</h2>
  <dl class="kv">
    <dt>状态判断</dt>
    <dd>{{ dft_tools_explain.status_judgement or "-" }}</dd>
    <dt>Provider / Model</dt>
    <dd>{{ dft_tools_explain.provider or "none" }} / {{ dft_tools_explain.model or "none" }}</dd>
  </dl>
  {% if dft_tools_explain.likely_causes %}
  <h3>主要原因</h3>
  <ul>
  {% for item in dft_tools_explain.likely_causes %}
    <li>{{ item }}</li>
  {% endfor %}
  </ul>
  {% endif %}
  {% if dft_tools_explain.next_actions %}
  <h3>下一步建议</h3>
  <ul>
  {% for item in dft_tools_explain.next_actions %}
    <li>{{ item }}</li>
  {% endfor %}
  </ul>
  {% endif %}
  <details>
    <summary>dft_tools explain 完整 JSON</summary>
    <pre>{{ dft_tools_explain_json }}</pre>
  </details>
</section>
{% endif %}
{% endblock %}
""",
    "job_status.html": """
{% extends "base.html" %}
{% block content %}
<h1 class="page-title">
  {% if back_url %}<a href="{{ back_url }}" class="back">&larr;</a>{% endif %}
  {{ title }}
</h1>

<section class="card">
  <dl class="kv">
    <dt>任务 ID</dt>
    <dd><span class="inline-code">{{ job.job_id }}</span></dd>
    <dt>状态</dt>
    <dd><span class="tag {{ job.status }}">{{ job.status }}</span></dd>
    <dt>创建时间</dt>
    <dd>{{ job.created_at }}</dd>
    <dt>开始时间</dt>
    <dd>{{ job.started_at or "-" }}</dd>
    <dt>完成时间</dt>
    <dd>{{ job.completed_at or "-" }}</dd>
    <dt>命令</dt>
    <dd><code class="inline-code">{{ command }}</code></dd>
  </dl>
  {% if job.status in ["queued","running"] %}
  <div class="notice">后台任务已提交。此页面 3 秒自动刷新一次，直到任务完成。</div>
  <script>
    setTimeout(function(){ window.location.reload(); }, 3000);
  </script>
  {% endif %}
</section>

<div class="split" style="margin-top:20px">
  <section class="card">
    <h3 style="margin-top:0">stdout</h3>
    <pre>{{ stdout or "(empty)" }}</pre>
  </section>
  <section class="card">
    <h3 style="margin-top:0">stderr</h3>
    <pre>{{ stderr or "(empty)" }}</pre>
  </section>
</div>
{% endblock %}
""",
}


def _bool_from_form(form: dict[str, list[str]], key: str) -> bool:
    return form.get(key, ["0"])[0] in {"1", "true", "on", "yes"}


def _first(form: dict[str, list[str]], key: str) -> str:
    return form.get(key, [""])[0].strip()


class SemiAutoDFTWebApp:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.store = RecordStore(self.project_root)
        self.templates = Environment(
            loader=DictLoader(TEMPLATES),
            autoescape=select_autoescape(default=True),
        )
        self.templates.filters["urlencode"] = lambda value: quote(str(value), safe="")

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
        setup_testing_defaults(environ)
        method = environ["REQUEST_METHOD"].upper()
        path = environ.get("PATH_INFO", "/")
        try:
            if method == "GET" and path == "/":
                return self._respond_html(start_response, self.render_home())
            if method == "GET" and path == "/runs":
                return self._respond_html(start_response, self.render_runs())
            if method == "GET" and path == "/run-detail":
                run_root = self._query_params(environ).get("run_root", [""])[0]
                return self._respond_html(start_response, self.render_run_detail(run_root))
            if method == "GET" and path == "/job-status":
                job_id = self._query_params(environ).get("job_id", [""])[0]
                return self._respond_html(start_response, self.render_job_status(job_id))
            if method == "POST" and path == "/actions/run":
                form = self._read_form(environ)
                return self._respond_html(start_response, self.handle_run_action(form))
            if method == "POST" and path == "/actions/step":
                form = self._read_form(environ)
                return self._respond_html(start_response, self.handle_step_action(form))
            if method == "POST" and path == "/actions/fetch":
                form = self._read_form(environ)
                return self._respond_html(start_response, self.handle_fetch_action(form))
            if method == "POST" and path == "/actions/dft-tools-explain":
                form = self._read_form(environ)
                return self._respond_html(start_response, self.handle_dft_tools_explain_action(form))
            if method == "POST" and path == "/actions/workflow":
                form = self._read_form(environ)
                return self._respond_html(start_response, self.handle_workflow_action(form))
            return self._respond_html(start_response, "<h1>404</h1><p>未找到页面。</p>", status="404 Not Found")
        except Exception as exc:  # pragma: no cover - fallback path
            return self._respond_html(
                start_response,
                self.templates.get_template("command_result.html").render(
                    title="Web UI 内部错误",
                    command="-",
                    returncode=1,
                    stdout="",
                    stderr=str(exc),
                    back_url="/",
                ),
                status="500 Internal Server Error",
            )

    def render_home(self) -> str:
        from dft_app.cluster_profiles import SUBMIT_PROFILES

        recent_runs = self.store.list_runs(limit=10)
        return self.templates.get_template("home.html").render(
            title="首页",
            active_nav="home",
            recent_runs=recent_runs,
            submit_profiles=sorted(SUBMIT_PROFILES.keys()),
        )

    def render_runs(self) -> str:
        return self.templates.get_template("runs.html").render(
            title="Runs 列表",
            active_nav="runs",
            runs=self.store.list_runs(limit=100),
        )

    def render_run_detail(self, run_root: str) -> str:
        resolved_root = self.store.resolve_run_root(run_root=run_root)
        view = load_run_detail_view(self.store, resolved_root)
        payload = view["payload"]
        return self.templates.get_template("run_detail.html").render(
            title=f"Run 详情 - {payload['run_record']['run_id']}",
            active_nav="runs",
            run_root=str(resolved_root),
            run_record=payload["run_record"],
            experiment_spec=payload.get("experiment_spec"),
            payload_json=json.dumps(payload, indent=2, ensure_ascii=False),
            workflow_status=view["workflow_status"],
            workflow_json=json.dumps(view["workflow_status"], indent=2, ensure_ascii=False) if view["workflow_status"] else "",
            candidate_state=view["candidate_state"],
            next_step_cards=view["next_step_cards"],
            user_flow_steps=view["user_flow_steps"],
            job_records=list_jobs(self.project_root, run_root=str(resolved_root), limit=10),
            explain_state=view["explain_state"],
            backflow_state=view["backflow_state"],
            dft_tools_explain=payload.get("dft_tools_explain_result"),
            dft_tools_explain_json=json.dumps(payload.get("dft_tools_explain_result"), indent=2, ensure_ascii=False) if payload.get("dft_tools_explain_result") else "",
            dft_tools_kb_ingest_result=payload.get("dft_tools_kb_ingest_result"),
            dft_tools_kb_ingest_json=json.dumps(payload.get("dft_tools_kb_ingest_result"), indent=2, ensure_ascii=False) if payload.get("dft_tools_kb_ingest_result") else "",
        )

    def render_job_status(self, job_id: str) -> str:
        job = load_job(self.project_root, job_id)
        stdout = self._read_optional_text(Path(job["stdout_path"]))
        stderr = self._read_optional_text(Path(job["stderr_path"]))
        return self.templates.get_template("job_status.html").render(
            title="后台任务状态",
            back_url=self._run_detail_url(job["run_root"]) if job.get("run_root") else "/",
            job=job,
            command=" ".join(job["command"]),
            stdout=stdout,
            stderr=stderr,
        )

    def handle_run_action(self, form: dict[str, list[str]]) -> str:
        args = ["run"]
        prompt = _first(form, "prompt")
        if prompt:
            args.append(prompt)
        for key, flag in (("material", "--material"), ("structure_path", "--structure-path"), ("submit_profile", "--submit-profile"), ("selected_candidate_id", "--selected-candidate-id")):
            value = _first(form, key)
            if value:
                args.extend([flag, value])
        if _bool_from_form(form, "dry_run"):
            args.append("--dry-run")
        if _bool_from_form(form, "submit"):
            args.append("--submit")
        if _bool_from_form(form, "remote"):
            args.append("--remote")
        return self._render_job_submission("后台执行 dft run", args, back_url="/")

    def handle_step_action(self, form: dict[str, list[str]]) -> str:
        run_root = _first(form, "run_root")
        phase = _first(form, "phase")
        args = ["step", phase, "--run-root", run_root]
        if _bool_from_form(form, "remote"):
            args.append("--remote")
        return self._render_job_submission("后台执行 dft step", args, back_url=self._run_detail_url(run_root), run_root=run_root)

    def handle_fetch_action(self, form: dict[str, list[str]]) -> str:
        run_root = _first(form, "run_root")
        args = ["fetch", "--run-root", run_root, "--run-id", self.store.load_run_record(Path(run_root)).run_id]
        return self._render_job_submission("后台执行 dft fetch", args, back_url=self._run_detail_url(run_root), run_root=run_root)

    def handle_dft_tools_explain_action(self, form: dict[str, list[str]]) -> str:
        run_root = _first(form, "run_root")
        args = ["dft-tools-explain", "--run-root", run_root]
        return self._render_job_submission("后台执行 dft_tools explain bridge", args, back_url=self._run_detail_url(run_root), run_root=run_root)

    def handle_workflow_action(self, form: dict[str, list[str]]) -> str:
        run_root = _first(form, "run_root")
        action = _first(form, "action")
        args = ["adsorption-workflow", "--run-root", run_root]
        if action == "status":
            args.append("--status")
        elif action == "submit":
            args.append("--submit")
        elif action == "monitor":
            args.append("--monitor")
        elif action == "fetch":
            args.append("--fetch")
        elif action == "parse-analyze":
            args.append("--parse-analyze")
        else:
            raise ValueError(f"不支持的 workflow action: {action}")
        return self._render_job_submission("后台执行 adsorption-workflow", args, back_url=self._run_detail_url(run_root), run_root=run_root)

    def _load_report_payload(self, run_root: Path) -> dict[str, Any]:
        record = self.store.load_run_record(run_root)
        return {
            "run_record": record.to_dict(),
            "experiment_spec": self.store.read_metadata_file(run_root, "experiment_spec.json"),
            "experiment_plan": self.store.read_metadata_file(run_root, "experiment_plan.json"),
            "build_summary": self.store.read_metadata_file(run_root, "build_summary.json"),
            "parsed_result": self.store.read_metadata_file(run_root, "parsed_result.json"),
            "analysis_summary": self.store.read_metadata_file(run_root, "analysis_summary.json"),
            "dft_tools_explain_result": self.store.read_metadata_file(run_root, "dft_tools_explain_result.json"),
            "dft_tools_knowledge_backflow_payload": self.store.read_metadata_file(run_root, "dft_tools_knowledge_backflow_payload.json"),
            "dft_tools_kb_ingest_result": self.store.read_metadata_file(run_root, "dft_tools_kb_ingest_result.json"),
            "adsorption_workflow_bundle": self.store.read_metadata_file(run_root, "adsorption_workflow_bundle.json"),
            "adsorption_workflow_status": self.store.read_metadata_file(run_root, "adsorption_workflow_status.json"),
        }

    def _render_command_result(self, title: str, args: list[str], back_url: str, run_root: str | None = None) -> str:
        result = self._run_cli(args)
        dft_tools_explain = None
        dft_tools_explain_json = ""
        if run_root:
            resolved_root = Path(run_root)
            dft_tools_explain = self.store.read_metadata_file(resolved_root, "dft_tools_explain_result.json")
            if dft_tools_explain is not None:
                dft_tools_explain_json = json.dumps(dft_tools_explain, indent=2, ensure_ascii=False)
            dft_tools_kb_ingest = self.store.read_metadata_file(resolved_root, "dft_tools_kb_ingest_result.json")
            dft_tools_kb_ingest_json = json.dumps(dft_tools_kb_ingest, indent=2, ensure_ascii=False) if dft_tools_kb_ingest is not None else ""
        else:
            dft_tools_kb_ingest = None
            dft_tools_kb_ingest_json = ""
        return self.templates.get_template("command_result.html").render(
            title=title,
            command=" ".join(result["command"]),
            returncode=result["returncode"],
            stdout=result["stdout"],
            stderr=result["stderr"],
            back_url=back_url,
            dft_tools_explain=dft_tools_explain,
            dft_tools_explain_json=dft_tools_explain_json,
            dft_tools_kb_ingest_result=dft_tools_kb_ingest,
            dft_tools_kb_ingest_json=dft_tools_kb_ingest_json,
        )

    def _render_job_submission(self, title: str, args: list[str], back_url: str, run_root: str | None = None) -> str:
        job = create_job(
            self.project_root,
            title=title,
            command=[sys.executable, "-m", "dft_app.cli.main", *args],
            run_root=run_root,
        )
        job["worker_pid"] = spawn_job_worker(self.project_root, job["job_id"])
        write_job(self.project_root, job)
        return self.render_job_status(job["job_id"])

    @staticmethod
    def _run_detail_url(run_root: str) -> str:
        return f"/run-detail?run_root={quote(str(run_root), safe='')}"

    def _run_cli(self, args: list[str]) -> dict[str, Any]:
        command = [sys.executable, "-m", "dft_app.cli.main", *args]
        process = subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "command": command,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }

    @staticmethod
    def _read_optional_text(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _query_params(environ: dict[str, Any]) -> dict[str, list[str]]:
        return parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)

    @staticmethod
    def _read_form(environ: dict[str, Any]) -> dict[str, list[str]]:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        raw = environ["wsgi.input"].read(length).decode("utf-8") if length else ""
        return parse_qs(raw, keep_blank_values=True)

    @staticmethod
    def _respond_html(start_response: Callable[..., Any], content: str, status: str = "200 OK") -> list[bytes]:
        start_response(status, [("Content-Type", "text/html; charset=utf-8")])
        return [content.encode("utf-8")]


def run_server(project_root: Path, host: str = "127.0.0.1", port: int = 8787) -> None:
    app = SemiAutoDFTWebApp(project_root)
    print(f"Web UI 已启动: http://{host}:{port}")
    with make_server(host, port, app) as server:
        server.serve_forever()
