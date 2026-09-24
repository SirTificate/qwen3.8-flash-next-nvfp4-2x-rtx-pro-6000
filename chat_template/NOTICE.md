# Chat template: origin, changes, license

`qwen38_flash_next.jinja` is derived from the `chat_template.jinja` shipped with
[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
at revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`, which in turn comes from
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

## Changes

1. **System messages in the middle of a conversation.** The original template raises
   `System message must be at the beginning.` for any system message that is not the first
   message. Claude Code sends a correction the user types during a running tool call as such a
   message, which then fails with HTTP 400. This template renders every further system message
   as its own system block at its position in the conversation.
2. **Effort aliases.** The original template accepts `reasoning_effort` values `xhigh`, `medium`
   and `low` and rejects everything else with HTTP 400. Clients commonly send `high` or
   `minimal`. This template maps `high` to `xhigh` and `minimal` to `low`.

Everything else is unchanged. The full diff against the original:

```diff
--- original/chat_template.jinja
+++ chat_template/qwen38_flash_next.jinja
@@ -45,6 +45,12 @@
 {%- set reasoning_instructions = '' %}
 {%- if enable_thinking is undefined or enable_thinking is true %}
     {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
+    {#- Derived: map the common client values 'high' -> 'xhigh' and 'minimal' -> 'low'. The original template rejects both with an error. -#}
+    {%- if resolved_reasoning_effort == 'high' %}
+        {%- set resolved_reasoning_effort = 'xhigh' %}
+    {%- elif resolved_reasoning_effort == 'minimal' %}
+        {%- set resolved_reasoning_effort = 'low' %}
+    {%- endif %}
     {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
         {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
     {%- endif %}
@@ -102,8 +108,9 @@
 {%- for message in messages %}
     {%- set content = render_content(message.content, true)|trim %}
     {%- if message.role == "system" %}
+        {#- Derived: the original template raises "System message must be at the beginning." here. Clients such as Claude Code send further system messages during a conversation; each one is rendered as its own system block at its position. The first system message is rendered in the preamble above. -#}
         {%- if not loop.first %}
-            {{- raise_exception('System message must be at the beginning.') }}
+            {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
         {%- endif %}
     {%- elif message.role == "user" %}
         {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
```

## License

The model and its original chat template are provided under the NVIDIA Open Model License
(https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) for the NVIDIA checkpoint and the Qwen Community License 1.0 for the
base model. The Qwen Community License requires that its notice is included with copies and
derivative works; it is reproduced below. Note its terms for commercial "Model as a Service"
and "AI Work Assistant" offerings.

Qwen Community License 1.0

Copyright (c) 2026 Qwen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software, including the model weights, parameters, configuration files, inference code and associated documentation files (collectively, the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, sell, deploy, host, fine-tune, and create derivative works from (collectively, "Use" or "Using") copies of the Software; and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. If the Software (or any derivative works thereof) is Used for any of the licensee's commercial products or services that have more than 100,000,000 monthly active users or US$ 20,000,000 (or equivalent in other currencies) monthly revenue, respective model name must be prominently displayed on the user interface of such product or service; and,

2. If the licensee or any of its affiliates conducts a Model as a Service or AI Work Assistant business, the licensee shall obtain a separate license from Qwen before Using the Software or its derivative works for any commercial purpose. The foregoing requirement shall not apply to the licensee's internal Use of the Software, provided that such Use does not make the Software, its outputs, or its underlying model capabilities available to any third party.

"Model as a Service" means giving a third party access to language model inference or fine-tuning (e.g., via API or a hosted endpoint) in a manner that allows such third parties to exercise meaningful control over the inputs, parameters, or training data. This does not include the mere relaying of requests to models hosted by other third parties. 
“AI Work Assistant” means an independent AI-powered product primarily designed for AI-assisted coding or office productivity (e.g., Qoder and QwenWork). It does not include: (a) a single-purpose AI tool (such as an AI translation tool); (b) an AI assistant primarily designed for a domain other than coding or office productivity (such as Taobao AI Shopping Assistant or AMap AI Chat); or (c) an AI assistant that is a feature of a product whose primary purpose is not AI-assisted coding or office productivity.

THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL QWEN, ITS AFFILIATES OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. THE USE OF THE SOFTWARE MUST COMPLY WITH APPLICABLE LAWS AND REGULATIONS, AND MUST NOT INFRINGE THE INTELLECTUAL PROPERTY RIGHTS OF ANY THIRD PARTY.

For any questions regarding this license, please contact model-business@notice.qwencloud.com.
