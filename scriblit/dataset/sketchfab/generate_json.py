import json

# your three base folder IDs
base_ids = [
    "0b5da073be88481091dbef7e55f1d180",
    "4e6688dcb7b34c36ba81c8303ed078d1",
    "907ac12c61744803b22a49efd74ec40a",
]

prompt_text = "a dark living room with a green louge chair, table and couch"
prompt_text_list = [
    "Industrial loft living–dining in deep shadow: white brick walls, exposed wood beams, and dim sconces. A charcoal sofa, white coffee table, and teal lounge chair sit on a muted patchwork rug. Built-in cabinets and a lone barstool. A rustic wooden table, metal stools, and a bicycle frame; a plant, flowers, and prints.",
    "Subdued industrial bathroom: terrazzo walls and exposed wood beams above hardwood floors. A long dark-stone vanity holds twin sinks beneath unlit mirrors, with a succulent and toiletry bottles between. To the right, a freestanding tub under a shuttered window and a potted plant. A deep-blue wave-pattern rug lies underfoot, faucets and countertop.",
    "Softly lit patient room with two white beds featuring orange rails under half-drawn blinds. A wooden nightstand holds a small lamp beside an IV pole and tray. Opposite, wood-paneled cabinets flank a recessed sink. A metal chair and sofa bench display plush toys."]

with open("~/scriblit/dataset/sketchfab/output.json", "w") as out:
    for prompt_i, base in enumerate(base_ids):
        for r in range(1, 61):
            folder = f"{base}_r_{r:02d}"
            for i in range(14):
                idx = f"{i:03d}"
                entry = {
                    "normal":  f"~/svd_relight/sketchfab/rendering_pp/{folder}/normals_{idx}.png",
                    "shading": f"~/svd_relight/sketchfab/rendering_pp/{folder}/mask_{idx}.png",
                    "albedo":  f"~/svd_relight/sketchfab/rendering_pp/{folder}/diffuse_{idx}.png",
                    "target":  f"~/svd_relight/sketchfab/rendering_pp/{folder}/colors_{idx}.png",
                    "prompt":  prompt_text_list[prompt_i]
                }
                out.write(json.dumps(entry) + "\n")
