<script>
    import * as Diff from 'diff';
    import { config1, config2 } from './dummy_config';
    import { slide } from 'svelte/transition';
    let c1 = config1;
    let c2 = config1;
    $: diff = Diff.diffLines(c1, c2, { newlineIsToken: true });
    $: lines = diff_to_lines(diff);
    function diff_to_lines(diff) {
        let lines = [];
        for (let part of diff) {
            part.value = part.value.trim();
            for (let line of part.value.split('\n')) {
                lines.push({ value: line, added: part.added, removed: part.removed });
            }
        }
        return lines;
    }

</script>

{@debug c1}
{@debug c2}
{@debug diff}
{@debug lines}

<button on:click={() => c2 = (c2 === config1) ? config2 : config1}>
    Toggle Config
</button>

<!-- {#each diff as part (part.value)}
    {#if part.value.trim()}
    <div class="part" class:lightgreen={part.added} class:lightred={part.removed}
        transition:slide|global={{duration: 4000, axis: 'y'}}
    >
        {part.value.trim()}
    </div>
    {/if}
{/each} -->
{#each lines as line (line.value)}
    <div class="part" class:lightgreen={line.added} class:lightred={line.removed}
        transition:slide|global={{duration: 200, axis: 'y'}}
    >
        {line.value.trim()}
    </div>
{/each}

<style>
    .part {
        white-space: pre;
        outline: 2px solid pink;
        font-size: 1.4rem;
    }
    .lightgreen {
        background-color: lightgreen;
        font-weight: bold;
    }
    .lightred {
        font-weight: bold;
        background-color: lightcoral;
        animation: color_red_in 200ms;
    }
    @keyframes color_red_in {
        from {
            background-color: lightgoldenrodyellow;
        }
        to {
            background-color: lightcoral;
        }
    }
</style>