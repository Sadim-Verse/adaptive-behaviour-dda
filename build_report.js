const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, PageBreak,
  Header, Footer, PageNumber, convertInchesToTwip
} = require("docx");
const fs = require("fs");

const PAGE_W = 12240, PAGE_H = 15840, MARGIN = 1440;
const USABLE = PAGE_W - 2 * MARGIN; // 9360

const NAVY = "1F3A5F", ORANGE = "B5560A", GREEN = "1E6B3C", RED = "A3151E", GRAY = "555555";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 } });
}
function h3(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_3, spacing: { before: 220, after: 100 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
}
function bullet(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    bullet: { level: 0 },
    spacing: { after: 80 },
  });
}
function majorFinding(text) {
  return new Paragraph({
    children: [new TextRun({ text: "Key Finding: ", bold: true, color: NAVY }), new TextRun({ text })],
    spacing: { before: 120, after: 240 },
    shading: { type: ShadingType.CLEAR, fill: "EAF1FB" },
  });
}

function cell(text, opts = {}) {
  const { bold = false, width, shade, color, align = AlignmentType.LEFT } = opts;
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text: String(text), bold, color })],
    })],
  });
}

function makeTable(headers, rows, widths) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((htext, i) => cell(htext, { bold: true, width: widths[i], shade: "1F3A5F", color: "FFFFFF" })),
  });
  const dataRows = rows.map((r, idx) => new TableRow({
    children: r.map((val, i) => cell(val, { width: widths[i], shade: idx % 2 === 1 ? "F2F2F2" : undefined })),
  }));
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [headerRow, ...dataRows],
  });
}

function figureImage(path, targetWidthDxa, captionText) {
  const dims = imageDims[path];
  const wIn = targetWidthDxa / 1440;
  const hIn = wIn * (dims.h / dims.w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(path),
        transformation: { width: Math.round(wIn * 96), height: Math.round(hIn * 96) },
      })],
      spacing: { before: 120, after: 80 },
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: captionText, italics: true, size: 20, color: GRAY })],
      spacing: { after: 280 },
    }),
  ];
}

const imageDims = {
  "images/figure1_architecture.png": { w: 1683, h: 1373 },
  "images/figure3_reward_stability_single_steady.png": { w: 1650, h: 825 },
  "images/figure3_reward_stability_single_spike.png": { w: 1650, h: 825 },
  "images/figure3_reward_stability_multi_steady.png": { w: 1650, h: 825 },
  "images/figure3_reward_stability_multi_spike.png": { w: 1650, h: 825 },
  "images/figure3_reward_stability_obstacle_minigrid_steady.png": { w: 1650, h: 825 },
  "images/figure3_reward_stability_obstacle_minigrid_spike.png": { w: 1650, h: 825 },
};

// ==========================================================================
// DOCUMENT CONTENT
// ==========================================================================

const titlePage = [
  new Paragraph({ spacing: { before: 2400 } }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Adaptive Behaviour DDA System", bold: true, size: 56, color: NAVY })],
    spacing: { after: 200 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Implementation Update Report", bold: true, size: 36, color: GRAY })],
    spacing: { after: 120 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Supervisor Priority Corrections, Verification & Results (Priorities 1\u20136)", size: 24, italics: true, color: GRAY })],
    spacing: { after: 800 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Sadu Ibraheem Afolabi", size: 24 })],
    spacing: { after: 60 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Matric No: 2022/41007", size: 22, color: GRAY })],
    spacing: { after: 60 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "B.Sc. Computer Science \u2014 Osun State University, Osogbo", size: 22, color: GRAY })],
    spacing: { after: 60 },
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Supervisor: Mr. John Adeniran", size: 22, color: GRAY })],
    spacing: { after: 400 },
  }),
  new Paragraph({ children: [new PageBreak()] }),
];

const execSummary = [
  h1("Executive Summary"),
  p("This report documents the corrections and improvements made to the Dynamic Difficulty Adjustment (DDA) framework in response to supervisor feedback, and supersedes the previously submitted update. It covers all six numbered priorities raised, plus a number of concrete implementation bugs discovered and fixed while addressing them \u2014 each is documented with before/after evidence rather than asserted."),
  p("In brief:"),
  bullet("Priority 1 \u2014 the weak, uncitable custom environment was replaced with a procedurally-generated MiniGrid-based environment (ObstacleMiniGridEnv), debugged extensively, and retrained with domain-randomized difficulty."),
  bullet("Priority 2 \u2014 the instability metric was found to be structurally broken (denominator diluted every result toward zero) and replaced with two real, verified metrics: reversal frequency and rate-of-change variance."),
  bullet("Priority 3 \u2014 Flow Zone bounds were formalized using one consistent rule (40th\u201370th percentile of a static, DDA-free baseline distribution) applied identically across all three environments."),
  bullet("Priority 4 \u2014 bootstrapped 95% confidence intervals were added to the main benchmark results and the delta-sensitivity analysis."),
  bullet("Priority 5 \u2014 deprioritized by mutual agreement in favour of higher-value fixes; noted as outstanding."),
  bullet("Priority 6 \u2014 the Adaptation Latency metric was found to have two separate real bugs (single-crossing recovery detection, and a non-recovery sentinel value contaminating averages) \u2014 both fixed and verified against real data."),
  bullet("In the course of this work, a further bug was discovered independently of the six priorities: ChaseEnv and MultiChaseEnv silently capped real difficulty at 6.0 regardless of the DDA Controller's intended 9.0 ceiling, meaning prior Single/Multi spike-mode results never actually tested the agents past that lower bound. This has been fixed and all affected results regenerated."),
  p("Figures 1 and 3 have been regenerated to reflect the corrected architecture and the corrected underlying data respectively."),
];

// ---- Priority sections -----------------------------------------------------

const prioritiesSection = [
  h1("Section A \u2014 Priorities Addressed"),

  h2("Priority 1 \u2014 Environment Replacement"),
  p("The original environments (ChaseEnv, MultiChaseEnv) were flagged as \u201cweak\u201d \u2014 not because they were too easy, but because they were custom, unpublished, and only randomized initial conditions on a fixed layout rather than performing genuine procedural generation. This meant no external validation of generalization was possible."),
  p("A new environment, ObstacleMiniGridEnv, was built on MiniGrid's procedural generation primitives, with a single normalized difficulty scalar d \u2208 [0,1] matching the existing set_difficulty() interface \u2014 so the DDA Controller, spike protocol, and evaluation pipeline required no changes to work against it."),
  p("Building this environment surfaced and required fixing a sequence of real bugs, each verified against actual data before being trusted:"),
  bullet("Hazards were tracked internally but never written into the render grid (invisible on screen)."),
  bullet("Hazard vertical spawn rows were deterministic (not random), and at high difficulty the hazard's horizontal homing matched the player's own move speed exactly \u2014 making evasion mathematically close to impossible. Fixed by randomizing spawn rows and decoupling horizontal homing cadence from vertical fall cadence."),
  bullet("Hazards could spawn adjacent to the player, causing unfair instant deaths (\u201cspawn-kills\u201d). Fixed by confining spawns to the top half of the grid."),
  bullet("Difficulty was non-monotonic: grid area grew quadratically while hazard count grew linearly, so increasing d sometimes made the environment easier, not harder \u2014 a direct violation of the DDA framework's core assumption. Diagnosed via systematic sweeps and fixed by decoupling grid size from difficulty scaling (with a controlled release valve at the top of the range)."),
  bullet("All training runs were found to have used a single fixed difficulty (d = 0.0, the easiest setting) for the entire training budget, producing a hard capability cliff (agents were reasonable at low difficulty and collapsed near-instantly at high difficulty). Fixed via domain-randomized training \u2014 every training episode now samples difficulty uniformly across [0,1] \u2014 while evaluation deliberately keeps difficulty fixed at a representative point for comparability across checkpoints. All three algorithms were retrained under this fix."),
  majorFinding("Under matched training conditions (domain-randomized, 6M steps, identical budget), PPO showed the steepest improvement slope and highest final performance, though with correspondingly higher variance; DDQN showed the most stable improvement of the three."),

  h2("Priority 2 \u2014 Real Instability Metric"),
  p("The original oscillation metric divided the count of direction changes by the total number of episodes \u2014 including episodes where the controller made no adjustment at all (because it was saturated at a bound, or already inside the Flow Zone). This diluted the metric toward zero regardless of how often the controller genuinely reversed direction when it did move, reproducing exactly the \u201ccontroller appears stable at all deltas\u201d problem the priority was raised to fix."),
  p("Replaced with two metrics, both verified against real cached speed-history data before being adopted:"),
  bullet("Reversal frequency \u2014 the fraction of nonzero speed changes that flip direction, using nonzero transitions as the denominator."),
  bullet("Rate-of-change variance \u2014 Var(\u0394d) across the full episode history, a metric named explicitly in the original supervisor feedback."),
  majorFinding("Verified real, environment-differentiated, delta-varying values from a genuinely broken flat 0.0000 baseline \u2014 e.g. Procedural's reversal frequency rises from 0.15 at \u03b4=0.05 to a clean 1.0000 (full bang-bang oscillation) at \u03b4=1.00. Separately, Single's DDQN controller was found to show a genuine, verified 0.0000 reversal frequency under steady conditions \u2014 not a metric artifact, but a real finding: its policy is strong enough that the controller only ever needs to push difficulty upward, never back off."),

  h2("Priority 3 \u2014 Flow Zone Formalization"),
  p("Flow Zone bounds are now calculated with one consistent rule across every environment: the 40th\u201370th percentile of a pooled STATIC baseline distribution (frozen policy, fixed difficulty, no DDA Controller active). Calibrating from DDA-active data would be circular \u2014 the controller would already be steering toward whatever bound was being measured \u2014 so a neutral, controller-free reference point was used instead."),
  bullet("Single: (296.0, 765.0) \u2014 calibrated at difficulty 6.0 (deliberately, not at the DDA ceiling of 9.0), chosen to push all three algorithms above their comfort zone without collapsing the two weaker algorithms' performance to near-zero, so a balanced three-algorithm comparison remains possible."),
  bullet("Multi: (245.6, 470.1)"),
  bullet("Procedural: (\u221274.5, \u221223.7) \u2014 negative bounds are mathematically correct, not an error: reward is +1/step survival, \u2212100 on collision, so a negative mean reward reflects dying within roughly 15\u201330 steps on average, an honest snapshot of current agent capability at this newer, harder environment."),

  h2("Priority 4 \u2014 Confidence Intervals"),
  p("Bootstrapped 95% confidence intervals (percentile method, 10,000 resamples) have been added to the main benchmark results (Table 3), the steady-vs-spike comparison (Table 8), and the delta-sensitivity analysis, computed across the 5 evaluation seeds per cell. A bootstrap approach was used rather than a normal-assumption interval because n = 5 seeds does not comfortably justify a normality assumption."),

  h2("Priority 5 \u2014 Multi-Environment Training Curves"),
  p("Deprioritized by mutual agreement in favour of the higher-value fixes above, given time constraints. Noted here as an explicitly outstanding item rather than silently dropped."),

  h2("Priority 6 \u2014 Adaptation Latency Variance"),
  p("Two separate, real bugs were found and fixed, verified against real evaluation data at each step:"),
  bullet("Recovery detection previously used a single threshold-crossing \u2014 the rolling-average reward touching 80% of its pre-spike peak once was enough to declare recovery. This is exactly the shape of bug that produces high-variance, bimodal latency in a stochastic environment: a brief noisy crossing gets reported as \u201crecovery\u201d when the agent immediately relapses. Fixed by requiring 5 consecutive episodes above threshold before recovery is declared."),
  bullet("Separately, when a seed never recovers within the 500-episode evaluation window, the pipeline recorded a literal value of 500 as if it were a genuine recovery duration, rather than a censored/missing marker. Averaging this sentinel directly into the mean badly distorted both the mean and the standard deviation \u2014 e.g. one real cell had per-seed latencies of [23, 29, 37, 25, 500], naively averaging to 122.8 \u00b1 210.9 when the honest statement is \u201c4 of 5 seeds recovered in 28.5 \u00b1 6.2 episodes; 1 of 5 never recovered.\u201d Fixed by reporting recovery rate as its own statistic and computing latency mean/std only over seeds that actually recovered."),
  majorFinding("The sustained-recovery fix alone did not fully resolve the originally-flagged high-variance problem when tested in isolation \u2014 this is itself informative: it is evidence for the supervisor's alternative hypothesis (genuine environmental stochasticity), not proof the detection bug was the sole cause. Both fixes are reported together in Table 8, with partial-recovery cells (e.g. 4/5, 3/5) flagged explicitly rather than hidden inside an average."),

  h2("An Additional Bug Found Outside the Six Priorities"),
  p("While investigating an unexpectedly small Single-environment Flow Zone violation under DDQN, ChaseEnv.set_difficulty() and MultiChaseEnv.set_difficulty() were both found to silently clip real difficulty to 6.0, regardless of the DDA Controller's own tracked ceiling of 9.0. This meant every prior spike-mode evaluation on Single and Multi forced the controller's internal variable to 9.0 while the environment itself never actually exceeded 6.0 \u2014 the spike was, in effect, frequently a near no-op."),
  p("Fixed by correcting the clip to 9.0 in both environment files. All affected Single and Multi evaluation data (steady mode, spike mode, and delta sensitivity) was regenerated. The effect was substantial and directly verifiable: Single's spike-mode adaptation latency changed from 10 episodes to 140 episodes once the environment was genuinely tested past its old artificial ceiling \u2014 confirming the fix, not just changing a number arbitrarily."),
];

// ---- Table 3 ---------------------------------------------------------------

const w3 = [1600, 2800, 2960]; // Algorithm | Mean Reward [95% CI] | FZC (%) [95% CI]
function table3For(envName, rows) {
  return [
    h3(envName),
    makeTable(["Algorithm", "Mean Reward [95% CI]", "FZC (%) [95% CI]"], rows, w3),
    new Paragraph({ spacing: { after: 200 } }),
  ];
}

const table3Section = [
  h1("Section B \u2014 Updated Results Tables"),
  h2("Table 3 \u2014 Main Benchmark Results (Steady Mode, 5-seed aggregate)"),
  p("Flow Zone bounds (see Priority 3): Single (296.0, 765.0) \u00b7 Multi (245.6, 470.1) \u00b7 Procedural (\u221274.5, \u221223.7)."),
  ...table3For("Single Environment", [
    ["DQN", "613.5 [599.9, 623.3]", "56.3% [52.8, 59.3]"],
    ["DDQN", "788.6 [779.3, 798.8]", "68.3% [62.7, 73.8]"],
    ["PPO", "678.8 [672.0, 688.0]", "78.9% [76.5, 81.3]"],
  ]),
  ...table3For("Multi Environment", [
    ["DQN", "361.1 [359.5, 362.7]", "76.3% [74.5, 78.0]"],
    ["DDQN", "355.9 [351.8, 359.8]", "78.5% [75.6, 80.8]"],
    ["PPO", "368.5 [365.4, 372.0]", "66.2% [60.3, 71.7]"],
  ]),
  ...table3For("Procedural Environment", [
    ["DQN", "\u221234.9 [\u221236.4, \u221233.2]", "54.5% [49.3, 58.9]"],
    ["DDQN", "\u221234.9 [\u221238.0, \u221232.1]", "68.4% [62.2, 73.7]"],
    ["PPO", "27.6 [19.1, 37.6]", "42.1% [34.4, 50.9]"],
  ]),
  majorFinding("All three environments' steady-mode results were reproduced byte-identically on re-run (compile_results.py reads only cached, deterministic data), and a manual single-seed spot-check fell within the reported 95% CI \u2014 both used as independent verification before these numbers were finalized."),
];

// ---- Table 8 ---------------------------------------------------------------

const w8 = [1500, 1300, 2400, 2200, 1960];
function table8For(envName, rows) {
  return [
    h3(envName),
    makeTable(["Algorithm", "Mode", "Mean Reward [95% CI]", "FZC (%) [95% CI]", "Latency (eps, recovered only)"], rows, w8),
    new Paragraph({ spacing: { after: 200 } }),
  ];
}

const table8Section = [
  h2("Table 8 \u2014 Steady vs Spike Comparison (All Algorithms, 5-seed aggregate, \u03b4=0.05)"),
  p("Expanded relative to the previous version, which covered only DDQN at a single seed. Adaptation Latency is computed only over seeds that recovered within the 500-episode window (Priority 6); a recovery rate below 5/5 is noted explicitly rather than hidden inside the average."),
  ...table8For("Single Environment", [
    ["DQN", "Steady", "613.5 [599.9, 623.3]", "56.3% [52.8, 59.3]", "N/A (no spike)"],
    ["DQN", "Spike", "547.6 [537.2, 556.6]", "61.5% [56.7, 66.2]", "106.0 \u00b1 103.3  (5/5)"],
    ["DDQN", "Steady", "788.6 [779.3, 798.8]", "68.3% [62.7, 73.8]", "N/A (no spike)"],
    ["DDQN", "Spike", "751.4 [738.5, 765.4]", "58.6% [53.6, 63.0]", "86.6 \u00b1 33.0  (5/5)"],
    ["PPO", "Steady", "678.8 [672.0, 688.0]", "78.9% [76.5, 81.3]", "N/A (no spike)"],
    ["PPO", "Spike", "608.9 [603.3, 614.4]", "68.6% [66.9, 70.3]", "104.6 \u00b1 45.2  (5/5)"],
  ]),
  ...table8For("Multi Environment", [
    ["DQN", "Steady", "361.1 [359.5, 362.7]", "76.3% [74.5, 78.0]", "N/A (no spike)"],
    ["DQN", "Spike", "297.3 [291.2, 302.7]", "59.4% [57.9, 60.9]", "151.2 \u00b1 15.1  (4/5)"],
    ["DDQN", "Steady", "355.9 [351.8, 359.8]", "78.5% [75.6, 80.8]", "N/A (no spike)"],
    ["DDQN", "Spike", "289.7 [285.6, 293.8]", "60.2% [57.4, 63.6]", "145.2 \u00b1 27.4  (5/5)"],
    ["PPO", "Steady", "368.5 [365.4, 372.0]", "66.2% [60.3, 71.7]", "N/A (no spike)"],
    ["PPO", "Spike", "298.5 [296.4, 300.6]", "48.4% [44.7, 51.7]", "129.2 \u00b1 30.4  (5/5)"],
  ]),
  ...table8For("Procedural Environment", [
    ["DQN", "Steady", "\u221234.9 [\u221236.4, \u221233.2]", "54.5% [49.3, 58.9]", "N/A (no spike)"],
    ["DQN", "Spike", "\u221235.6 [\u221237.9, \u221233.4]", "52.5% [47.9, 56.4]", "23.3 \u00b1 5.9  (3/5)"],
    ["DDQN", "Steady", "\u221234.9 [\u221238.0, \u221232.1]", "68.4% [62.2, 73.7]", "N/A (no spike)"],
    ["DDQN", "Spike", "\u221236.5 [\u221240.0, \u221233.1]", "70.5% [66.2, 74.3]", "75.4 \u00b1 64.7  (5/5)"],
    ["PPO", "Steady", "27.6 [19.1, 37.6]", "42.1% [34.4, 50.9]", "N/A (no spike)"],
    ["PPO", "Spike", "26.7 [22.5, 30.8]", "43.5% [37.0, 51.2]", "56.2 \u00b1 67.0  (5/5)"],
  ]),
  majorFinding("Single's DQN and PPO show full 5/5 recovery despite high individual-episode variance in latency; Multi's DQN and Procedural's DQN show partial recovery (4/5, 3/5) \u2014 these partial-recovery cases are direct, concrete evidence for the environmental-stochasticity explanation of Priority 6's originally-flagged variance."),
];

// ---- Figures ---------------------------------------------------------------

const figuresSection = [
  h1("Section C \u2014 Updated Figures"),
  h2("Figure 1 \u2014 Full System Architecture Diagram"),
  p("Regenerated to reflect all three environments and all three algorithms (previously showed only ChaseEnv / DDQN), to remove a stale hardcoded Flow Zone bound, to correctly show the Perturbation Module overriding the Environment directly (not the Agent, as an earlier draft incorrectly implied), and to add the timescale labels, injection-point marker, and DDA update-rule annotation requested by the supervisor. Every claim on this diagram, including the exact update-rule formula, was checked line-by-line against the actual controller code before being finalized."),
  ...figureImage("images/figure1_architecture.png", 8800, "Figure 1: Full system architecture, verified against source code, showing the inner (per-timestep) and outer (per-episode-boundary) loops, the perturbation injection point, and the DDA update rule."),

  h2("Figure 3 \u2014 Reward Stability Under Dynamic Difficulty"),
  p("Regenerated for all three environments in both steady and spike mode, using corrected Flow Zone bounds and (for Single/Multi) the corrected 9.0 difficulty ceiling. Legend spacing and the Procedural reward axis (previously compressed and difficult to read against rare large early-episode outliers) were also fixed."),
  h3("Single Environment"),
  ...figureImage("images/figure3_reward_stability_single_steady.png", 8800, "Figure 3a: Single \u2014 Steady Mode."),
  ...figureImage("images/figure3_reward_stability_single_spike.png", 8800, "Figure 3b: Single \u2014 Spike Mode. Adaptation latency 140 episodes, following the difficulty-ceiling fix (previously 10 episodes under the uncorrected 6.0 clip)."),
  h3("Multi Environment"),
  ...figureImage("images/figure3_reward_stability_multi_steady.png", 8800, "Figure 3c: Multi \u2014 Steady Mode."),
  ...figureImage("images/figure3_reward_stability_multi_spike.png", 8800, "Figure 3d: Multi \u2014 Spike Mode. Adaptation latency 138 episodes."),
  h3("Procedural Environment"),
  ...figureImage("images/figure3_reward_stability_obstacle_minigrid_steady.png", 8800, "Figure 3e: Procedural \u2014 Steady Mode."),
  ...figureImage("images/figure3_reward_stability_obstacle_minigrid_spike.png", 8800, "Figure 3f: Procedural \u2014 Spike Mode. Adaptation latency 31 episodes."),
];

// ---- Limitations / files ----------------------------------------------------

const limitationsSection = [
  h1("Section D \u2014 Outstanding Items and Known Limitations"),
  bullet("Priority 5 (multi-environment training curves) remains outstanding, deprioritized by agreement."),
  bullet("Priority 4's confidence intervals have been added to Tables 3 and 8 and the delta-sensitivity analysis, but not yet propagated to every remaining table in the wider results set."),
  bullet("Single's Flow Zone was deliberately calibrated at a static difficulty of 6.0 rather than the DDA ceiling of 9.0, specifically to keep a balanced three-algorithm comparison rather than collapsing DQN and PPO's performance \u2014 a documented design choice, not an oversight."),
  bullet("Figures 2 and 4\u201310 from the original report have not been regenerated in this update cycle; this report covers Figures 1 and 3 only, which were the two explicitly requested by the supervisor for revision."),

  h1("Section E \u2014 Files Updated in This Cycle"),
  bullet("envs/obstacle_minigrid_env.py \u2014 procedurally-generated environment, domain-randomized training support."),
  bullet("envs/chase_env.py, envs/multi_chase_env.py \u2014 difficulty ceiling clip corrected from 6.0 to 9.0."),
  bullet("training/train_multi_env.py \u2014 sustained-recovery detection (Priority 6), corrected Flow Zone bounds."),
  bullet("scripts/compile_results.py \u2014 rewritten: corrected Flow Zone import, corrected folder paths, added Procedural, sentinel-safe latency aggregation, bootstrap CIs."),
  bullet("scripts/run_delta_sensitivity.py \u2014 corrected instability metrics, corrected Flow Zone import, bootstrap CIs, sentinel-safe latency aggregation."),
  bullet("scripts/recompute_adaptation_latency.py \u2014 new utility to reapply the Priority 6 fix to already-collected data without re-running expensive evaluations."),
  bullet("scripts/generate_figure3_reward_stability.py \u2014 legend spacing and reward-axis scaling fixes."),
  bullet("Architecture diagram (Figure 1) \u2014 fully revised and verified against source code."),
];

// ==========================================================================
// ASSEMBLE
// ==========================================================================

const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
      },
    },
    children: [
      ...titlePage,
      ...execSummary,
      ...prioritiesSection,
      ...table3Section,
      ...table8Section,
      ...figuresSection,
      ...limitationsSection,
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("Sadu_updated_v3.docx", buffer);
  console.log("Document written: Sadu_updated_v3.docx");
});