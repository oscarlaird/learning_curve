<script>
    import hljs from "highlight.js/lib/core";
    import python from "highlight.js/lib/languages/python";
    import 'highlight.js/styles/default.css'; 
    import { onMount } from "svelte";
    import * as Diff from 'diff';
    export let c1;
    export let c2;

    function diff_to_lines(diff) {
        let ls = [];
        for (let part of diff) {
            // part.value = part.value.trim();
            // part.value = part.value;
            // remove trailing newline
            part.value = part.value.replace(/\n$/, '');
            part.value = part.value.replace(/\n$/, '');
            // remove leading newline
            part.value = part.value.replace(/^\n/, '');
            for (let l of part.value.split('\n')) {
                ls.push({ value: l, added: part.added, removed: part.removed });
            }
        }
        return ls;
    }

    let diff = Diff.diffLines(c1, c2, { newlineIsToken: true });
    let lines = diff_to_lines(diff);

    onMount(() => {
        // Register language and highlight each line individually
        hljs.registerLanguage("python", python);

        document.querySelectorAll("code.code").forEach(element => {
           hljs.highlightElement(element); 
        });
    });
</script>

{#each lines as line}
    <code class="code language-python"
        class:added={line.added}
        class:removed={line.removed}
    >
        {line.value}
    </code>
{/each}

<style>
    code {
        display: block;
        font-size: 1.0rem;
        background-color: white;
        white-space: pre;
    }
    .added {
        background-color: lightgreen;
    }
    .removed {
        background-color: lightcoral;
    }
</style>
