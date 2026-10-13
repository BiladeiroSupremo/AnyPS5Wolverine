export class OutputDirectoryTransaction {
  #state = "open";

  constructor(parent, name, directory) {
    this.parent = parent;
    this.name = name;
    this.directory = directory;
  }

  static async create(parent, name) {
    try {
      await parent.getDirectoryHandle(name);
    } catch (error) {
      if (error.name !== "NotFoundError") throw error;
      const directory = await parent.getDirectoryHandle(name, { create: true });
      return new OutputDirectoryTransaction(parent, name, directory);
    }
    throw new Error(`The folder ${name} already exists. Choose another destination or rename the existing folder.`);
  }

  commit() {
    if (this.#state !== "open") throw new Error("Output directory transaction is no longer active");
    this.#state = "committed";
  }

  async discard() {
    if (this.#state !== "open") return;
    await this.parent.removeEntry(this.name, { recursive: true });
    this.#state = "discarded";
  }
}
