<script>
    import { fade } from "svelte/transition";
    import { training_mode, current_node_hash, proposed_node_hash } from "./stores";
    $: show_proposed = $proposed_node_hash !== null;
    const denoise_levels = [500, 999];

</script>

{#if (!$training_mode)}
<div class="samples_container"
    in:fade={{duration: 200}}
>
    <div class="samples_group">
        <img src={`/vertices/${$current_node_hash}/noisy_img_999.png`} alt="noisy_img_999" />
        <img src={`/vertices/${$current_node_hash}/denoised_img_999.png`} alt="denoised_img_999" />
    </div>
    <div class="samples_group">
        <img src={`/vertices/${$current_node_hash}/noisy_img_500.png`} alt="noisy_img_500" />
        <img src={`/vertices/${$current_node_hash}/denoised_img_500.png`} alt="denoised_img_500" />
    </div>
</div>
{:else}
<div class="sampling_text">
    <p>Sampling...</p>
</div>
{/if}

<style>
    .samples_container, .samples_group {
        display: flex;
        flex-direction: column;
    }
    .samples_container {
        height: 100%;
        justify-content: space-around;
    }
    img {
        height: 100px;
        width: auto;
        margin: 0 auto;
    }
    .sampling_text {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    .sampling_text p {
        font-size: 2rem;
        padding: 30px;
        border: 2px solid black;
    }
</style>