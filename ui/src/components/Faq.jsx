const QUESTIONS = [
  [
    "What does MAP-ExPLoc predict?",
    "The website’s Random Forest predicts one of five annotation classes: Cytoplasm, Membrane, Mitochondrion, Nucleus, or Secreted. It uses 423 sequence descriptors, including amino-acid and dipeptide frequencies, length, hydrophobicity, and isoelectric point. The output is a model prediction, not an observation of where a protein is located.",
  ],
  [
    "Which sequences can I submit?",
    "Use amino-acid sequences, not DNA or RNA. Paste a single sequence or upload FASTA with a unique identifier for each protein. Batch mode accepts up to 25 sequences and 50,000 residues in total. Only the 20 standard amino-acid letters are supported; ambiguous letters, gaps, and stop symbols are rejected. Check the sequence rather than silently deleting or replacing unsupported residues.",
  ],
  [
    "Can I use it for proteins from any species?",
    "The website’s model was trained on curated human proteins. Performance on other species has not been established. A sequence passing input validation does not mean the model is suitable for that organism, a short fragment, or an engineered construct.",
  ],
  [
    "Is the highest score the probability that the prediction is correct?",
    "No. The scores are uncalibrated model outputs. A score of 0.8 should not be read as an 80% chance of correct localization. Compare the full score distribution, check the model’s evaluation, and seek supporting evidence. A clear top score can still be wrong.",
  ],
  [
    "Can a protein have more than one location?",
    "Yes, but this model assigns one curated annotation class. Its single-label output cannot establish exclusive localization or resolve changes across cell types, conditions, or isoforms. The five classes also do not cover every cellular compartment, so the top class may be an incomplete description.",
  ],
  [
    "Do the explanation plots identify targeting signals?",
    "Not on their own. The website’s SHAP plots show how engineered sequence features contribute to the model output relative to its reference. Positive and negative contributions raise or lower that output. Whole-sequence features cannot identify a particular residue or motif as a targeting signal, and an attribution does not establish a biological mechanism.",
  ],
  [
    "How well has the model been validated?",
    "The served Random Forest achieved a macro-F1 of 0.5223 on its original 364-protein holdout. Macro-F1 averages F1 across classes; it is not the fraction of proteins predicted correctly. Performance differs by class, with weak Cytoplasm recall. Separate internal model-selection results do not validate this served model, and independent transfer remains unestablished.",
  ],
  [
    "How should I use these results?",
    "Use them to generate hypotheses and prioritize follow-up. Compare predictions with curated annotations and published evidence for the exact protein and isoform. Localization claims need appropriate experimental support; these predictions and explanations do not supply it. For reproducibility, retain the input sequence, model metadata, and exported report.",
  ],
];

export default function Faq() {
  return (
    <section className="faq" aria-labelledby="faq-heading">
      <div className="faq-heading">
        <h2 id="faq-heading">Common questions</h2>
        <a href="https://github.com/zeyuyaoy/mapexploc/blob/stable/docs/index.md">
          Methods and evaluation
        </a>
      </div>
      <div className="faq-questions">
        {QUESTIONS.map(([question, answer]) => (
          <details key={question}>
            <summary>{question}</summary>
            <p>{answer}</p>
          </details>
        ))}
      </div>
    </section>
  );
}
