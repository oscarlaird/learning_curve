<script>
    import { onMount } from "svelte";
    export let layer;
    export let old_layer;
    const height_scale_factor = 40;
    const width_per_channel = 1;
    import { tweened } from "svelte/motion";
    import { fade } from "svelte/transition";
    function channels_to_width(channels) {
        let x = Math.log(channels) / Math.log(2);
        return x * x * width_per_channel;
    }
    function size_to_height(size) {
        // size^0.5 * height_scale_factor
        return height_scale_factor * Math.pow(size, 0.7);
    }
    let init_width = channels_to_width(layer.channels);
    let tweened_width = tweened(init_width, {duration: 200});
    $: tweened_width.set(channels_to_width(layer.channels));
    $: height = size_to_height(layer.size);
</script>

<div class="feature_map_box" 
    class:changed={old_layer && old_layer.channels !== layer.channels}
    style:height={height}px
    style:width={$tweened_width}px
>
    <div class="channels_text"
        class:changed={old_layer && old_layer.channels !== layer.channels}
    >
        {layer.channels}
    </div>
    <div class="height_text"
        class:changed={old_layer && old_layer.size !== layer.size}
    >
        {layer.size} × {layer.size}
    </div>
</div>


<style>
    .feature_map_box {
        margin-top: 30px;
        border: 2px solid black;
        position: relative;
        min-width: 30px;
        background: linear-gradient(90deg, white, lightgray);
    }
    .feature_map_box.changed {
        background: linear-gradient(90deg, white, lightblue);
    }
    .channels_text {
        position: absolute;
        bottom: 100%;
        left: 0;
        width: 100%;
        display: flex;
        justify-content: center;
        font-size: 1.6rem;
    }
    .height_text {
        position: absolute;
        left: 0;
        top: 50%;
        display: flex;
        justify-content: center;
        transform: translate(-50%, -50%) rotate(-90deg) translateY(50%);
        background-color: white;
        border: 1px solid black;
        outline: 1px dashed pink;
        padding: 0 10px;
        text-wrap: nowrap;
        font-size: 1.4rem;
    }
    .channels_text.changed {
        color: blue;
        transform: scale(1.2);
    }
</style>

