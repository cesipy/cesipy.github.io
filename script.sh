pandoc posts/bachelor_thesis.md \
  --from markdown+raw_html+tex_math_dollars+tex_math_single_backslash-yaml_metadata_block \
  --template=template.html \
  --standalone \
  --metadata-file=/dev/null \
  --mathjax \
  -o posts/bachelor-thesis.html