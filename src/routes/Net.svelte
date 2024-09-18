<script>
    import { current_node_hash, proposed_node_hash, training_mode } from "./stores";
    let current_layers = [];
    let proposed_layers = [];
    // refetch the current and proposed layers when the current/proposed hash changes
    $: show_proposed = proposed_layers.length > 0;
    $: layers = show_proposed ? proposed_layers : current_layers;
    //
    function get_old_layer_by_id(id) {
        return current_layers.find(layer => layer.id === id) ?? null;
    }
    //
    import FeatureMap from "./FeatureMap.svelte";
    import Module from "./Module.svelte";
    import { flip } from 'svelte/animate';
    import { fade } from 'svelte/transition';
    import { slide } from 'svelte/transition';
    import { onMount } from 'svelte';

    onMount(() => {
        current_node_hash.subscribe( async (hash) => {
            if (hash === null) {
                current_layers = [];
                return;
            }
            fetch(`vertices/${hash}/net_arch.json`)
                .then(response => response.json())
                .then(data => {
                    current_layers = data;
            });
        });
        proposed_node_hash.subscribe( async (hash) => {
            if (hash === null) {
                proposed_layers = [];
                return;
            }
            fetch(`vertices/${hash}/net_arch.json`)
                .then(response => response.json())
                .then(data => {
                    proposed_layers = data;
            });
        });
    })
</script>

{@debug layers}

<div class="net_container">
{#each layers as layer (layer.id)}
    <!-- <div in:fade={{duration: 600}} out:fade={{duration: 600}} animate:flip={{duration: 600}} class="layer_container"> -->
    <div transition:slide={{axis: 'x', duration: 200}} class="layer_container">
        {#if layer.type === "module"}
            <Module {layer} old_layer={get_old_layer_by_id(layer.id)} />
        {:else if layer.type === "feature_map"}
            <FeatureMap {layer} old_layer={get_old_layer_by_id(layer.id)} />
        {/if}
    </div>
{/each}
</div>

<style>
    .net_container {
        display: flex;
        flex-direction: row;
        align-items: end;
        justify-content: center;
    }
    .layer_container {
    }
</style>