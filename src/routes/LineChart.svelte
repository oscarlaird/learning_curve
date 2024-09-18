<!-- LineChart.svelte -->
<!-- Adapted from the wonderful svelte/tailwind for dataviz site skeleton -->
<script>
    import { training_mode, current_node_hash, proposed_node_hash } from './stores';
    import { writable } from 'svelte/store';
    import { scaleTime, scaleLinear } from 'd3-scale';
    import { extent, max } from 'd3-array';
    import { line, curveBasis } from 'd3-shape';
    import Axis from './Axis.svelte';
    import Labels from './Labels.svelte';
    import { csv } from 'd3-fetch';
    import { onMount } from 'svelte';
    import { draw } from 'svelte/transition';
    import { cubicIn, expoIn, linear, quadIn } from 'svelte/easing';
    import { parse } from 'svelte/compiler';
  
    let data, xScale, yScale;
  
    $: show_proposed = $proposed_node_hash !== null;
    let shown_hash = writable($current_node_hash);
    $: shown_hash.set(show_proposed ? $proposed_node_hash : $current_node_hash);
    // Get data
    // let's use csv function from d3-fetch to download our data.
    // download data on: https://datavisualizationwithsvelte.com/data/natural_gas.csv
    onMount(() => {
      shown_hash.subscribe((hash) => {
        csv(`vertices/${hash}/losses.csv`).then((csv) => (data = csv));
      });
      training_mode.subscribe((mode) => {
        if (!mode) return;  // don't do anything if not in training mode
        console.log('training mode TIMEOUT T-6s');
        setTimeout(() => {
          $current_node_hash = $proposed_node_hash;
          $proposed_node_hash = null;
          $training_mode = false;
        }, 2000);
      });
    });
  
    // Dimensions, Margins, Scales
    let width; // width will be set by the clientWidth
    const height = 450;
    const margin = { top: 10, right: 10, bottom: 20, left: 35 };
  
    $: if (data && width) {
      xScale = scaleLinear()
        .domain(extent(data, (d) => d.epochs))
        .range([margin.left, width - margin.right]);
  
      yScale = scaleLinear()
        .domain(extent(data, (d) => +d.losses))
        .range([height - margin.bottom, margin.top]);
    }

    $: final_loss = data ? data[data.length - 1].losses : null;
  
    // Line function from d3 to create the d attribute for a path element
    // which will be our line,
    $: lineGenerator = line()
      .x((d) => xScale(d.epochs))
      .y((d) => yScale(+d.losses))
      .curve(curveBasis);
  </script>
  
  <!-- bind width of the container div to the svg width-->
  <!-- what this will to is to set the width of the svg responsively, same width like its container div -->
  <div class="wrapper" bind:clientWidth={width}>
    {#if final_loss}
      <div class="final_loss">
        {!show_proposed ? "Final loss: " + parseFloat(final_loss).toFixed(3) : ($training_mode ? "Training..." : "Final loss: ???")}
      </div>
    {/if}

    {#if data && width}
      <svg {width} {height}>
        <Axis
          {width}
          {height}
          {margin}
          tick_number={width > 380 ? 10 : 4}
          scale={xScale}
          position="bottom" />
        <Axis {width} {height} {margin} scale={yScale} position="left" />
        <Labels
          labelfory={true}
          {width}
          {height}
          {margin}
          tick_number={10}
          yoffset={-50}
          xoffset={270}
          label={'Log Loss'} />
        {#key data}
          {#if !show_proposed || $training_mode}
          <path
            in:draw={{ duration: 2000, easing: linear }}
            shape-rendering="crispEdges"
            d={lineGenerator(data)}
            stroke="black"
            stroke-width={1.5}
            stroke-linecap="round"
            fill="none" />
          {/if}
        {/key}
      </svg>
    {/if}
</div>

<style>
  .final_loss {
    position: absolute;
    top: 70px;
    right: 70px;
    background-color: white;
    height: 150px;
    width: 350px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 2rem;
    border: 2px solid black;
  }
</style>