<script>
    import * as THREE from 'three'
    import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
    let graph;
    import { onMount } from 'svelte';
    import { training_mode, current_node_hash, proposed_node_hash } from './stores.js';
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const scene = new THREE.Scene();
    scene.background = null;
    const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    const big_sphere_geometry = new THREE.SphereGeometry(1., 32, 32);
    const small_sphere_geometry = new THREE.SphereGeometry(0.6, 32, 32);
    // revealed nodes in red
    const material_revealed = new THREE.MeshStandardMaterial({
        color: 0xff0000,
        roughness: 0.5,
        metalness: 0.5
    })
    // adjacent nodes in white
    const material_white = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.5,
        metalness: 0.5
    })
    // other nodes in grey
    const material_other = new THREE.MeshStandardMaterial({
        color: 0xaaaaaa,
        roughness: 0.5,
        metalness: 0.5
    })
    const material_adjacent = new THREE.MeshStandardMaterial({
        color: 0x00ff00,
        roughness: 0.5,
        metalness: 0.5
    })

    const ball_objects = [];
    let hover_ball = null;
    let vertices = [];
    let edges = [];

    function onMouseMove(event) {
        // disable during training
        if ($training_mode) return;
        const rect = graph.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1; // -1 to 1
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1; // -1 to 1
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(ball_objects);
        if (intersects.length > 0) {
            let intersected_ball = intersects[0].object;
            // make the ball bigger on hover
            intersected_ball.geometry.dispose();
            intersected_ball.geometry = big_sphere_geometry;
            hover_ball = intersected_ball;


            let intersected_ball_idx = ball_objects.indexOf(intersected_ball);
            let vertex = vertices[intersected_ball_idx];
            // propose the node on hover
            let hash = vertex.id;
            if (hash !== $current_node_hash) {
                $proposed_node_hash = hash;
            }
        } else {
            // unselect the node on un-hover
            $proposed_node_hash = null;
            // return sphere to normal size
            if (hover_ball) {
                hover_ball.geometry.dispose();
                hover_ball.geometry = small_sphere_geometry;
                hover_ball = null;
            }
        }
    }

    function onClick(event) {
        if ($proposed_node_hash) {
            $training_mode = true;
        }
    }

    function set_revealed_material(hash) {
        console.log(hash);
        ball_objects.forEach((ball, idx) => {
            if (vertices[idx].id === hash) {
                ball.material.dispose();
                ball.material = material_revealed;
                // set neighbors to white (unless already revealed)
                for (let i = 0; i < vertices.length; i++) {
                    if (edges[idx][i] && vertices[i].id !== $current_node_hash) {
                        ball_objects[i].material.dispose();
                        ball_objects[i].material = material_adjacent;
                    }
                }
            }
        });
    }

    $: set_revealed_material($current_node_hash);

    onMount(async () => {
        // synchronously fetch the balls from vertices.json
        vertices = await fetch('vertices.json').then(response => response.json());
        edges = await fetch('edges.json').then(response => response.json());
        console.log(1);
        // create a basic barbell
        const renderer = new THREE.WebGLRenderer({ canvas: graph, antialias: true, alpha: true });
        renderer.setSize(1000, 1000);
        camera.position.z = 22;
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.screenSpacePanning = true;
        controls.enablePan = true;
        

        vertices.forEach(vertex => {
            console.log(vertex.id, $current_node_hash, vertex.id === $current_node_hash);
            let mat = (vertex.id === $current_node_hash) ? material_revealed : material_other;
            const sphere = new THREE.Mesh(small_sphere_geometry, mat);
            sphere.position.set(vertex.pos[0], vertex.pos[1], vertex.pos[2]);
            ball_objects.push(sphere);
            scene.add(sphere);
        });

        for (let i = 0; i < vertices.length - 1; i++) {
            for (let j = i + 1; j < vertices.length; j++) {
                if (!edges[i][j]) continue;
                const midpoint = new THREE.Vector3().addVectors(ball_objects[i].position, ball_objects[j].position).divideScalar(2);
                const direction = new THREE.Vector3().subVectors(ball_objects[j].position, ball_objects[i].position);
                const height = direction.length();
                const geometry = new THREE.CylinderGeometry(0.1, 0.1, height, 32);
                const cylinder = new THREE.Mesh(geometry, material_white);
                cylinder.position.copy(midpoint);
                cylinder.lookAt(ball_objects[j].position);
                cylinder.rotateX(Math.PI / 2);
                scene.add(cylinder);
            }
        }
        // lighting
        // key light
        const light = new THREE.DirectionalLight(0xffffff, 1.5);
        light.position.set(0, 10, 10);
        scene.add(light);
        // ambient light
        const ambient_light = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambient_light);
        // back light
        const back_light = new THREE.DirectionalLight(0xffffff, 1);
        back_light.position.set(0, 0, -10);
        scene.add(back_light);



        // animate
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        // indicate root node
        console.log('current_node_hash', $current_node_hash);
        set_revealed_material($current_node_hash);
        console.log(3);
    });

</script>

<canvas class="graph" on:mousemove={onMouseMove} on:click={onClick}
    bind:this={graph}
></canvas>

<style>
    .graph {
        width: 1000px;
        height: 1000px;
        outline: 3px dashed black;
    }
</style>