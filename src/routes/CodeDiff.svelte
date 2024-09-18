<script>
    import { onMount } from "svelte";
    import { training_mode, current_node_hash, proposed_node_hash } from "./stores";
    import CodeDiffInner from "./CodeDiffInner.svelte";
    let current_code = '\n';
    let proposed_code = '\n';
    $: show_proposed = $proposed_node_hash !== null;
    $: c1 = current_code;  // we always compare to the current code
    $: c2 = show_proposed ? proposed_code : current_code;


    onMount(() => {
        current_node_hash.subscribe((hash) => {
            if (hash === null) return;
            fetch(`vertices/${hash}/net.py`)
                .then(response => response.text())
                .then(text => {
                    current_code = text;
                });
        });
        proposed_node_hash.subscribe((hash) => {
            if (hash === null) return;
            fetch(`vertices/${hash}/net.py`)
                .then(response => response.text())
                .then(text => {
                    proposed_code = text;
                });
        });
    });
</script>

{#key c1 }
{#key c2 }
    <CodeDiffInner {c1} {c2} />
{/key}
{/key}