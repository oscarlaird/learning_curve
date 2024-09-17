<script>
    export let layers;
    import FeatureMap from "./FeatureMap.svelte";
    import Module from "./Module.svelte";
    import { flip } from 'svelte/animate';
    import { fade } from 'svelte/transition';
    import { slide } from 'svelte/transition';
</script>

<div class="net_container">
{#each layers as layer (layer.id)}
    <!-- <div in:fade={{duration: 600}} out:fade={{duration: 600}} animate:flip={{duration: 600}} class="layer_container"> -->
    <div transition:slide={{axis: 'x'}} class="layer_container">
        {#if layer.type === "module"}
            <Module {layer} />
        {:else if layer.type === "feature_map"}
            <FeatureMap {layer} />
        {/if}
    </div>
{/each}
</div>

<style>
    .net_container {
        display: flex;
        flex-direction: row;
        align-items: end;
        height: 500px;
    }
    .layer_container {
        outline: 1px dashed black;
    }
</style>