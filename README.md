# Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucinations in Large Vision-Language Models
<figure>
  <img src="assets/MoD_main.png" width="100%">
  <figcaption>
    <strong>Overview of our proposed MoD.</strong> 
    MoD involves three key steps: (1) extracting the model's attended image tokens while masking the others; (2) generating vanilla output logits from both original image tokens and masked image tokens; and (3) computing the JS divergence between the two logit distributions to assess the correctness of the model's attention. Based on this evaluation, MoD adaptively adopts either complementary or contrastive decoding strategies to produce hallucination-free outputs. The upper and lower panels illustrate cases of right and wrong attention, respectively. The corresponding attended image, visualized using LLaVA-1.5, is displayed in the lower left corner of each panel.
  </figcaption>
</figure>
