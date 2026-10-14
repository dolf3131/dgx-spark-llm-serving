# tensorfold-qwen38:v0.3.6.3 + xgrammar for response_format / guided_* / structured_outputs ('tensorfold[grammar]' asks for xgrammar>=0.2.8,<0.3).
# --no-deps on purpose: xgrammar lists torch, and pip would replace the image's NGC torch with a PyPI build.
FROM tensorfold-qwen38:v0.3.6.3
RUN pip install --no-cache-dir --no-deps 'xgrammar==0.2.8' 'apache-tvm-ffi>=0.1.11' \
 && python3 -c "import torch, transformers, xgrammar, importlib.metadata as m; print('torch', torch.__version__, '| transformers', transformers.__version__, '| xgrammar', m.version('xgrammar'), '| tvm-ffi', m.version('apache-tvm-ffi'))"
