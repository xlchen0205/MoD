import argparse
import os
import torch
import json
from tqdm import tqdm
import shortuuid
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from Qwen_VL.modeling_qwen import QWenLMHeadModel
from transformers import set_seed,AutoTokenizer,AutoModelForCausalLM

from PIL import Image
import math

from MoD_utils.vcd_add_noise import add_diffusion_noise
from MoD_utils.MoD_sampling import evolve_MoD_sampling
evolve_MoD_sampling()

def recorder(out):
    NEG_WORDS = ["No", "not", "no", "NO"]
    out = out.replace('.', '')
    out = out.replace(',', '')
    words = out.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if args.model_name == "qwen-vl":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id
        model = QWenLMHeadModel.from_pretrained(
            model_path,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_idx"]
        image_file = line["image_path"]
        qs = line["question"]
        GT = line["answer"]
        category = line["category"]
        qs = qs.split('\n')[0] 
        cur_prompt = qs
        if args.model_name == "llava-1.5":
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        elif args.model_name == "qwen-vl":
            image_path = os.path.join(args.image_folder, image_file)
            question = '<img>{}</img>{} Answer:'.format(image_path, qs)
            questions_id = []
            input_ids = tokenizer([question], return_tensors='pt', padding='longest')
        elif args.model_name == "llava-next":
            prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n<|question|> ASSISTANT:"
            prompt = prompt.replace('<|question|>', qs)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=False,
                use_cd=args.use_cd,
                use_avisc=args.use_avisc,
                use_m3id=args.use_m3id,
                use_JS_consis=args.use_JS_consis,
                JS_PH_alpha=args.JS_PH_alpha,
                JS_MH_alpha=args.JS_MH_alpha,
                JS_beta=args.JS_beta,
                JS_top_attn=args.JS_top_attn,
                JS_threshold=args.JS_threshold,
                layer_gamma=args.layer_gamma,
                lamb=args.lamb,
                model_name=args.model_name,
            )
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            if model_name == "qwen-vl":
                outputs = [tokenizer.decode(_[input_ids.input_ids.size(1):].cpu(), skip_special_tokens=True).strip() for _ in pred][0]
            else:
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
        outputs = recorder(outputs)

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "GT": GT,
                                   "model_outputs": outputs,
                                   "category": category,
                                   "image": image_file,
                                   "model_id": model_name,}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="path_to/llava-v1.5-7b/")
    parser.add_argument("--model_name", type=str, default="llava-1.5", choices=["llava-1.5", "qwen-vl", "llava-next"])
    parser.add_argument("--model-base", type=str, default=None)
    
    parser.add_argument("--image-folder", type=str, default="./data/image")
    parser.add_argument("--question-file", type=str, default=".data/MME_question.jsonl")
    parser.add_argument("--answers-file", type=str, default="output/llava-1.5_MME.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)

    parser.add_argument("--JS_PH_alpha", type=float, default=4.0)
    parser.add_argument("--JS_MH_alpha", type=float, default=1.0)
    parser.add_argument("--JS_beta", type=float, default=0.5)
    parser.add_argument("--JS_top_attn", type=float, default=0.20)
    parser.add_argument("--JS_threshold", type=float, default=0.05)

    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_avisc", action='store_true', default=False)
    parser.add_argument("--use_m3id", action='store_true', default=False)
    parser.add_argument("--use_JS_consis", action='store_true', default=True)

    parser.add_argument("--layer_gamma", type=float, default=0.5) 
    parser.add_argument("--lamb", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
