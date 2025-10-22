pandoc posts/research-j.md \
  --from markdown+raw_html+tex_math_dollars-yaml_metadata_block \
  --template=template.html \
  --standalone \
  --metadata-file=/dev/null \
  -o posts/20251021-research-j.html