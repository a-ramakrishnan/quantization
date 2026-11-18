type Tree = { code: number } | { left: Tree; right: Tree };

function packBits(bits: string) {
  const bytes = new Uint8Array(Math.ceil(bits.length / 8));
  for (let index = 0; index < bits.length; index += 8) {
    bytes[index / 8] = parseInt(bits.slice(index, index + 8).padEnd(8, "0"), 2);
  }
  return bytes;
}

function bitReader(bytes: Uint8Array) {
  let position = 0;
  return () => {
    if (position >= bytes.length * 8) throw new Error("Unexpected end of prefix stream");
    const bit = (bytes[Math.floor(position / 8)]! >> (7 - position % 8)) & 1;
    position += 1;
    return bit;
  };
}

/** Preorder tree format: a branch is 0, a leaf is 1 followed by a signed INT4 code. */
export function encodeInt4Prefix(codes: number[]) {
  const counts = new Map<number, number>();
  for (const code of codes) {
    if (!Number.isInteger(code) || code < -8 || code > 7) throw new Error("Expected signed INT4 codes");
    counts.set(code, (counts.get(code) ?? 0) + 1);
  }
  const queue = [...counts].sort(([a], [b]) => a - b).map(([code, count]) => ({ count, tree: { code } as Tree }));
  while (queue.length > 1) {
    queue.sort((a, b) => a.count - b.count);
    const [left, right] = queue.splice(0, 2);
    queue.push({ count: left!.count + right!.count, tree: { left: left!.tree, right: right!.tree } });
  }

  const paths = new Map<number, string>();
  function serialize(tree: Tree, path: string): string {
    if ("code" in tree) {
      paths.set(tree.code, path);
      return "1" + (tree.code & 15).toString(2).padStart(4, "0");
    }
    return "0" + serialize(tree.left, path + "0") + serialize(tree.right, path + "1");
  }
  const treeBits = queue[0] ? serialize(queue[0].tree, "") : "";
  const words = codes.map(code => paths.get(code)!);
  const payloadBits = words.join("");
  const symbols = [...counts].map(([code, count]) => ({ code, count, path: paths.get(code)! }))
    .sort((a, b) => b.count - a.count || a.code - b.code);
  return { treeBits, payloadBits, words, symbols, treeBytes: packBits(treeBits), payloadBytes: packBits(payloadBits) };
}

export function decodeInt4Prefix(treeBytes: Uint8Array, payloadBytes: Uint8Array, count: number) {
  if (count === 0) return [];
  const readTreeBit = bitReader(treeBytes);
  function readTree(): Tree {
    if (readTreeBit() === 0) return { left: readTree(), right: readTree() };
    let nibble = 0;
    for (let index = 0; index < 4; index++) nibble = nibble * 2 + readTreeBit();
    return { code: nibble >= 8 ? nibble - 16 : nibble };
  }
  const root = readTree();
  const readPayloadBit = bitReader(payloadBytes);
  return Array.from({ length: count }, () => {
    let node = root;
    while (!("code" in node)) node = readPayloadBit() === 0 ? node.left : node.right;
    return node.code;
  });
}
