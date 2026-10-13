import assert from "node:assert/strict";
import { test } from "node:test";
import { OutputDirectoryTransaction } from "../output-directory.mjs";

function missingDirectory() {
  const error = new Error("not found");
  error.name = "NotFoundError";
  return error;
}

test("creates a new output root only after confirming the destination is absent", async () => {
  const directory = {};
  const calls = [];
  const parent = {
    async getDirectoryHandle(name, options) {
      calls.push([name, options]);
      if (!options) throw missingDirectory();
      return directory;
    },
    async removeEntry() {}
  };

  const transaction = await OutputDirectoryTransaction.create(parent, "app-linux");

  assert.equal(transaction.directory, directory);
  assert.deepEqual(calls, [["app-linux", undefined], ["app-linux", { create: true }]]);
});

test("does not claim or delete a destination that already exists", async () => {
  const parent = {
    async getDirectoryHandle() { return {}; },
    async removeEntry() { assert.fail("existing destination must not be removed"); }
  };

  await assert.rejects(
    OutputDirectoryTransaction.create(parent, "app-linux"),
    /already exists/u
  );
});

test("discards its incomplete output recursively", async () => {
  let removed;
  const parent = {
    async getDirectoryHandle(name, options) {
      if (!options) throw missingDirectory();
      return { name };
    },
    async removeEntry(...args) { removed = args; }
  };
  const transaction = await OutputDirectoryTransaction.create(parent, "app-linux");

  await transaction.discard();
  await transaction.discard();

  assert.deepEqual(removed, ["app-linux", { recursive: true }]);
});

test("keeps completed output and reports cleanup failures", async () => {
  let removeCount = 0;
  const parent = {
    async getDirectoryHandle(name, options) {
      if (!options) throw missingDirectory();
      return { name };
    },
    async removeEntry() {
      removeCount++;
      throw new Error("permission denied");
    }
  };
  const completed = await OutputDirectoryTransaction.create(parent, "complete-linux");
  completed.commit();
  await completed.discard();
  assert.equal(removeCount, 0);

  const failed = await OutputDirectoryTransaction.create(parent, "partial-linux");
  await assert.rejects(failed.discard(), /permission denied/u);
  assert.equal(removeCount, 1);
});
