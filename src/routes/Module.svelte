<script>
    export let layer;
    import { fade } from 'svelte/transition';
    import { expoOut } from 'svelte/easing';
</script>

<div class="module_box">
    <div class="title_text"
    >
        {layer.title}
    </div>
    <!-- other properties (exlcuding title, id, type) -->
    {#each Object.keys(layer).filter(key => key !== "title" && key !== "id" && key !== "type") as key}
        <div class="module_property" 
        >
            {key}: {layer[key]}
        </div>
    {/each}
    <div class="triangle"></div>
</div>

<style>
    :root {
        --h: 100px;
    }

    .module_box {
        border: 2px solid black;
        border-radius: 8px;
        height: var(--h);
        position: relative;
        margin-left: 10px;
        margin-right: 30px;
        margin-bottom: 20px;
        margin-top: 30px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        width: 60px;
        text-overflow: clip;
    }
    .title_text {
        position: absolute;
        bottom: 100%;
        font-size: 1rem;
        text-wrap: nowrap;
        z-index: 2;
    }
    .module_property {
        font-size: 0.9rem;
        text-wrap: nowrap;
        z-index: 2;

    }
    .triangle {
        position: absolute;
        top: 0;
        left: 100%;
        width: 0;
        /* height: 100%; */
        border-left: 20px solid black;
        border-bottom: calc(var(--h)/2.0) solid transparent;
        border-top: calc(var(--h)/2.0) solid transparent;
    }
    .triangle::after {
        position: absolute;
        top: calc(-1 * var(--h)/2.0);
        left: -22px;
        content: '';
        width: 0;
        border-left: 20px solid white;
        border-bottom: calc(var(--h)/2.0) solid transparent;
        border-top: calc(var(--h)/2.0) solid transparent;
    }



</style>