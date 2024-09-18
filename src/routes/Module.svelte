<script>
    export let layer;
    export let old_layer;
    import { fade } from 'svelte/transition';
    import { expoOut } from 'svelte/easing';
    import { get } from 'svelte/store';
</script>

<div class="module_box"
    class:added={old_layer === null || old_layer.title !== layer.title}
>
    <div class="title_text"
        class:added={old_layer === null}
        class:changed={old_layer && old_layer.title !== layer.title}
    >
        {layer.title}
    </div>
    <!-- other properties (exlcuding title, id, type) -->
    {#each Object.keys(layer).filter(key => key !== "title" && key !== "id" && key !== "type") as key}
        <div class="module_property"
            class:added={old_layer === null || old_layer && !(key in old_layer)}
            class:changed={old_layer && (key in old_layer) && old_layer[key] !== layer[key]}
        >
            {key}: {layer[key]}
        </div>
    {/each}
    <div class="triangle"></div>
</div>

<style>
    :root {
        --h: 130px;
    }

    .module_box {
        border: 2px solid black;
        border-radius: 6px;
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
        width: 100px;
        text-overflow: clip;
        background: linear-gradient(90deg, white, lightgray);
    }
    .module_box.added {
        color: green;
        border-color: green;
    }
    .title_text {
        position: absolute;
        bottom: 100%;
        font-size: 1.6rem;
        text-wrap: nowrap;
        z-index: 2;
    }
    .module_property {
        font-size: 0.9rem;
        text-wrap: nowrap;
        font-size: 1.4rem;
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
        border-left: 20px solid lightgrey;
        border-bottom: calc(var(--h)/2.0) solid transparent;
        border-top: calc(var(--h)/2.0) solid transparent;
    }
    .module_box.added .triangle {
        border-left-color: green;
    }


</style>