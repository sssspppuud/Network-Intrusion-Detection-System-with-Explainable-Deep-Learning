# ECM3401 Final Report — Section 5 (Reflection & Conclusion) + Abstract
# Ralph Lickley | IoMT IDS with XAI
# Deadline: 12pm, 29 April 2026
---
> Paste directly into your LaTeX source. No placeholders — everything
> is written from your actual results. Minor grammar/typo fixes to existing
> sections are noted at the bottom.

---

## ABSTRACT

### Integrating Explainable AI with Deep Learning for Intrusion Detection in IoMT Networks

The Internet of Medical Things (IoMT) is increasingly integrated into modern
healthcare delivery, enabling continuous remote patient monitoring and data-driven
clinical decision-making. Alongside its benefits, this growing connectivity has
substantially expanded the cyberattack surface facing healthcare infrastructure,
with hundreds of significant data breaches recorded annually, raising serious
concerns for patient data privacy and safety. Machine Learning (ML) based
Intrusion Detection Systems (IDS) have demonstrated strong performance in
detecting malicious network traffic; however, the opacity of complex deep learning
models limits their adoption in safety-critical healthcare environments where
clinicians must be able to understand and trust the system's decisions.

This project addresses this gap by designing, implementing, and evaluating a Deep
Neural Network (DNN) IDS trained on the CIC2024 IoMT dataset — a purpose-built
benchmark simulating 40 IoMT devices across 18 attack types, including IoMT-specific
protocols such as MQTT and Bluetooth. A two-stage sampling pipeline combining SMOTE
oversampling and random undersampling was applied to mitigate a pronounced class
imbalance prior to training. The trained DNN achieved a macro average F1-score of
0.8603 across five traffic categories, exceeding the project target of 0.80 and
substantially outperforming the baseline DNN reported for this dataset (F1 = 0.665).

SHAP (Shapley Additive exPlanations) was applied to 1,000 stratified test samples
to provide global and per-class feature importance explanations. Analysis of the
SHAP outputs confirms that the model's classifications are grounded in features
consistent with known network-level attack signatures — for example, high SRate and
ICMP packet counts drive DDoS predictions, while elevated fin\_count is the primary
indicator for Recon. These results demonstrate that a transparent and high-performing
IDS for IoMT environments is achievable, providing a foundation for trustworthy
ML-based network security in healthcare settings.

> Word count: ~230. Trim or expand by one sentence if needed to hit 200–250.

---

## SECTION 5 — Reflection and Conclusion

---

### 5.1 Critical Reflection

**Strengths.**

The project successfully delivered all four primary objectives defined in Section 1.3.
The DNN IDS achieved a macro average F1-score of 0.8603 on the held-out test set,
comfortably exceeding the target of 0.80 and demonstrating reliable multi-class
classification across the five traffic categories. Performance was particularly strong
on MQTT (F1 = 0.9933), Benign (F1 = 0.9752), and Recon (F1 = 0.9697), demonstrating
that the model is highly effective at identifying three of the five classes with near
clinical-grade reliability. A key strength of these results is how substantially they
improve upon the benchmark DNN from the original CIC2024 dataset paper, which
achieved an F1-score of 0.665 on a comparable classification task [11]. The
improvement of 0.195 F1 points is largely attributable to the resampling pipeline
introduced in this project — an approach the benchmark paper does not employ — which
gave the model a more balanced view of the traffic distribution during training and
improved its ability to generalise across minority classes.

The decision to drop the Spoofing class, while a departure from the original six-class
framing, was a principled and well-reasoned choice. With Spoofing comprising only 0.2%
of the dataset, any strategy to retain it would have involved either extreme
oversampling — risking the generation of synthetic samples that do not reflect real
spoofing traffic patterns — or discarding large volumes of majority class data. The
trade-off was correctly identified early, and the report transparently acknowledges
the resulting detection gap, which reflects good engineering judgement.

The pivot from dual SHAP/LIME analysis to a SHAP-only approach was similarly
well-justified. LIME's instability under repeated runs — observed empirically during
testing and consistent with the known sensitivity of its random perturbation sampling
— would have produced unreliable explanations that undermine the project's core goal
of making model decisions trustworthy. Focusing on the depth of SHAP analysis rather
than breadth across two methods yielded richer, more reliable results and demonstrates
an ability to adapt the project scope based on evidence rather than adhering rigidly
to the original plan.

The SHAP analysis itself constitutes a genuine intellectual contribution. The per-class
beeswarm plots reveal that the model has learned attack-specific feature patterns that
are broadly consistent with established network security knowledge: high SRate and
ICMP/TCP packet counts for DDoS, elevated fin\_count and IAT for Recon, high
ack\_flag\_number and TCP reliance for MQTT, and high packet Weight with large IAT
for Benign traffic. This level of agreement between the model's learned behaviour and
domain knowledge provides meaningful assurance that the DNN is classifying traffic
for the right reasons, rather than exploiting spurious correlations in the training data.

**Weaknesses and limitations.**

The most significant limitation of the model is its performance on the DoS class,
where it achieved an F1-score of only 0.5350. The confusion matrix reveals the scale
of this problem clearly: 105,727 DoS samples were misclassified as DDoS, and 96,726
DDoS samples were misclassified as DoS, making the DoS/DDoS boundary by far the
dominant source of error in the model. The SHAP analysis in Section 4.3.2 directly
illuminates why: the DoS class beeswarm plot (Fig. 10) shows the model relying on
SRate and syn\_count — the same features it uses to identify DDoS — but in an
inverted direction, with high SRate values pushing the model \textit{away} from DoS
rather than toward it. This is not the expected signature for DoS attacks, which
should be primarily characterised by a sustained high-rate connection from a single
source. The core problem is that the CIC2024 IoMT dataset's feature set does not
include source and destination IP addresses, which would allow the model to
distinguish between the distributed multi-source traffic of DDoS and the
single-source pattern of DoS. This is a fundamental limitation of the available
features, not a failure of the model architecture, but it is an important caveat
for any practical deployment. An IDS with a 47.5\% false negative rate for DoS
would provide limited protection against this attack type in a real healthcare
network, where a successful DoS attack could disrupt the delivery of treatment
through IoMT devices.

A second limitation is the reliance on synthetic training data. SMOTE generates
minority-class samples by interpolating between real examples in feature space,
which addresses the numerical imbalance but does not perfectly replicate the
statistical properties of genuine network traffic. For classes like Recon and MQTT —
where performance is strong — this risk appears to be well-managed. However, the
strong SMOTE performance numbers for these classes should be interpreted with
some caution, as the model may have learned a slightly smoothed version of their
feature distributions rather than the full natural variability found in real IoMT
networks.

A third limitation concerns the scope of the XAI evaluation. The validity of the
SHAP explanations is assessed qualitatively, by comparing feature attributions against
network security domain knowledge. While this is standard practice in the XAI
literature — no formal ground-truth explanation benchmark exists for this dataset —
it means the explanations cannot be validated quantitatively. A clinician or security
engineer deploying this system in practice would need to conduct additional validation,
for example by consulting with network security domain experts who can confirm or
challenge specific attributions.

Finally, the SHAP computation, applied to 1,000 test instances with a background
dataset drawn from the training set, is not suitable for real-time deployment in its
current form. The current XAI integration is most useful as an offline model
validation and audit tool, rather than a live explanation system that could support
real-time clinical decision-making.

---

### 5.2 Conclusion

This project set out to address the dual challenges of detection performance and
interpretability in IoMT intrusion detection. A DNN IDS was designed, implemented,
and evaluated on the CIC2024 IoMT dataset, achieving a macro average F1-score of
0.8603 across five traffic categories — exceeding the 0.80 target and substantially
improving upon the benchmark DNN from the original dataset paper (0.665). SHAP-based
XAI analysis was successfully integrated and confirmed that the model's classification
decisions are grounded in features consistent with known network attack signatures
across four of the five classes, enhancing the trustworthiness of the system for
use in safety-critical healthcare settings.

The primary contribution of this project is the combination of a high-performing DNN
IDS with rigorous SHAP-based XAI evaluation on a purpose-built IoMT dataset —
addressing a gap in the literature where XAI techniques have rarely been applied in
IoMT-specific contexts [10, 11]. The results demonstrate that strong classification
performance and meaningful interpretability can be achieved simultaneously, rather
than treating them as competing objectives. The SHAP analysis also provides a
concrete diagnostic tool: the DoS/DDoS confusion identified through the beeswarm
plots directly informs what additional features (specifically, source IP diversity)
would most improve the model in future iterations — a finding that would not have
been accessible from performance metrics alone.

Reviewing the four objectives from Section 1.3: the dataset preparation objective
was fully met, with the two-stage resampling pipeline successfully reducing the
maximum-to-minimum class ratio from approximately 44:1 to 4:1 in the training set.
The DNN implementation objective was fully met, with the F1 target exceeded. The
XAI integration objective was fully met for SHAP, with LIME de-scoped on principled
empirical grounds. The interpretability validation objective was substantially met,
with the caveat that qualitative validation carries inherent limitations and the
DoS class explanations revealed a feature-level limitation rather than confirming
the model's correctness for that class.

---

### 5.3 Future Directions

Three concrete directions for extending this work are proposed, each motivated
directly by a limitation or finding identified in this project.

**1. Incorporating IP-based features to resolve the DoS/DDoS boundary.**
The most immediately actionable improvement to the model would be the inclusion of
source and destination IP address diversity features — for example, the number of
unique source IPs per time window. The SHAP analysis in Section 4.3.2 showed that
the model currently cannot reliably distinguish DoS (single-source, sustained) from
DDoS (multi-source, distributed) because the available feature set does not encode
this information. Adding flow-level source diversity metrics, which could be derived
from the raw packet captures underlying the CIC2024 IoMT dataset, would provide the
model with the discriminating signal it currently lacks and is likely to substantially
improve DoS recall from its current 0.5242. This is a targeted, evidence-based
enhancement rather than a generic suggestion, and it could be implemented without
retraining the entire pipeline from scratch.

**2. Addressing the Spoofing class with generative modelling.**
The removal of the Spoofing class (0.2\% of the dataset) means the deployed system
would be unable to detect ARP spoofing attacks — a meaningful gap given that MITM
and ARP spoofing are realistic attack vectors in the Fog Layer of IoMT networks.
Future work should investigate class-conditional generative approaches, such as
Conditional Variational Autoencoders (CVAEs) or Conditional GANs trained to
synthesise realistic Spoofing-class traffic, rather than SMOTE interpolation which
is ill-suited to such severe imbalances. Recent work in network traffic generation
suggests that generative models can produce synthetic flows that are statistically
closer to real traffic than SMOTE samples for minority classes, making this a
promising avenue for retaining the full attack taxonomy in future iterations.

**3. Federated learning for privacy-preserving model improvement.**
A fundamental constraint of this project — and of all benchmark-trained IDS models —
is the dataset-to-deployment gap: the model was trained on simulated lab traffic and
its performance on traffic from genuinely novel IoMT device types is unknown.
Improving the model on real-world data requires access to real IoMT network traffic,
which raises significant patient data privacy concerns. Federated learning offers
a principled solution: local IDS models could be trained on-device within individual
hospital networks, with only model weight updates (rather than raw traffic data)
shared with a central aggregation server. This would allow the model to learn from
diverse, real-world IoMT traffic without centralising sensitive data — directly
addressing the ethical constraint noted in Section 2.3.2 that motivates the use
of a synthetic dataset in this project.

---

## QUICK FIXES FOR EARLIER SECTIONS
(Minor corrections spotted while reading your draft — fix these before submission)

1. **Section 3.1, line 1:** "traing\_test\_split" → \texttt{train\_test\_split}
   (typo in the LaTeX \texttt{} block)

2. **Section 4.1, line 4:** "more than able tot distinguish" → "more than able to
   distinguish" (extra 't')

3. **Section 4.2, title:** The section discusses only the benchmark paper comparison,
   not a Random Forest baseline. If you don't have time to add an RF baseline, rename
   this subsection "Benchmark Comparison" to accurately describe its content and
   avoid the examiner noting its absence.

4. **Figure 5 caption:** Currently reads "Loss and accuracy curves for training and
   validation data" — this is the SHAP global summary plot, not the loss curves.
   Change to: "Global SHAP summary plot showing mean absolute SHAP values for
   the top 10 features across all predictions."

5. **Section 1.1 background subsections:** These take up roughly 1.5 pages alongside
   the introduction, leaving very little room. Check your total intro + background
   stays within the 2-page limit — if it overruns, condense §1.1.2 and §1.1.3 as
   these are the most literature-review-like parts and least suited to the final report.

6. **Abstract placeholder:** The title page still shows placeholder text. Write the
   abstract above into the title page before submission.

---
*End of file. Deadline: 12pm, 29 April 2026.*
