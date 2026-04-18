#!/usr/bin/env bash
# archive/ 配下の各 subdir を tar.zst に圧縮、元 subdir を削除。
# 可逆: zstd -d < xxx.tar.zst | tar x で復元可。
set -e
cd "$(dirname "$0")/.."

ARCHIVE="models/checkpoints/archive"
cd "$ARCHIVE"

for d in */; do
    name="${d%/}"
    if [ -f "$name.tar.zst" ]; then
        echo "  skip $name (already compressed)"
        continue
    fi
    echo "=== compress $name ==="
    tar c "$name" 2>/dev/null | zstd --long -19 -T0 > "$name.tar.zst"
    orig=$(du -sb "$name" | cut -f1)
    comp=$(stat -c%s "$name.tar.zst")
    ratio=$(awk "BEGIN{printf \"%.1f\", $orig/$comp}")
    echo "  $name: $(du -sh $name | cut -f1) -> $(du -sh $name.tar.zst | cut -f1) (${ratio}x)"
    rm -rf "$name"
done

echo
echo "=== archive final ==="
du -sh .
ls -la *.tar.zst | head
