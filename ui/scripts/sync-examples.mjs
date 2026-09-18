import {readFile, writeFile} from "node:fs/promises";

const source = new URL(
    "../../src/mapexploc/examples/proteins.json",
    import.meta.url,
);
const records = JSON.parse(await readFile(source, "utf8"));
const outputs = [
    [
        new URL("../src/examples.json", import.meta.url),
        JSON.stringify(records, null, 2) + "\n",
    ],
    [
        new URL("human_examples.fasta", source),
        records
            .map(({id, name, sequence}) => `>${id} ${name}\n${sequence}\n`)
            .join(""),
    ],
];
for (const [target, expected] of outputs) {
    if (process.argv.includes("--check")) {
        if ((await readFile(target, "utf8")) !== expected) {
            throw new Error(
                `Examples are stale: ${target.pathname}. Run pnpm examples:sync.`,
            );
        }
    } else {
        await writeFile(target, expected);
    }
}
