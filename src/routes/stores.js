import { writable } from "svelte/store";


const root_node_hash = "ffa28f73d2152125e17ac9528bcc6940fa8c8b04";

const training_mode = writable(false);
const current_node_hash = writable(root_node_hash);
const proposed_node_hash = writable(null);


export { training_mode, current_node_hash, proposed_node_hash }