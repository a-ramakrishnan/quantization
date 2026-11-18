import { describe, expect, it } from "vitest";
import { decodeInt4Prefix, encodeInt4Prefix } from "./prefix-compression";

describe("packed INT4 prefix coding", () => {
  it("encodes the 8/4/2/2 example in 28 bits and decodes the padded bytes", () => {
    const codes = [7, 7, -8, 7, 0, 7, -8, 7, -1, 7, -8, 7, 0, 7, -8, -1];
    const encoded = encodeInt4Prefix(codes);
    expect(encoded.payloadBits).toHaveLength(28);
    expect(encoded.payloadBytes).toHaveLength(4);
    expect(encoded.treeBits).toHaveLength(23);
    expect(encoded.treeBytes).toHaveLength(3);
    expect(decodeInt4Prefix(encoded.treeBytes, encoded.payloadBytes, codes.length)).toEqual(codes);
  });

  it("uses no payload bits when the tree and count determine every code", () => {
    const codes = [-8, -8, -8];
    const encoded = encodeInt4Prefix(codes);
    expect(encoded.payloadBytes).toHaveLength(0);
    expect(decodeInt4Prefix(encoded.treeBytes, encoded.payloadBytes, codes.length)).toEqual(codes);
  });

  it("keeps an empty block empty", () => {
    const encoded = encodeInt4Prefix([]);
    expect(encoded.treeBytes).toHaveLength(0);
    expect(decodeInt4Prefix(encoded.treeBytes, encoded.payloadBytes, 0)).toEqual([]);
  });
});
