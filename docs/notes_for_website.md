# Notes for the Website – figure captions in plain language

Figures live in `figures/`. Each note below is written for non-technical readers and can be pasted directly under the image.

## Data & Prep tab

### `01_class_distribution.png` – Tickets per support group
Our dataset contains nearly 48,000 real support requests, each already assigned to one of eight support teams. The teams are very unequal in workload: Hardware and HR Support together receive more than half of all requests, while Administrative rights gets fewer than 4 in 100. This matters because a system that only learns the busy teams well could still look accurate overall while failing the smaller teams, so we measure performance for every team separately.

### `02_ticket_length_hist.png` – How long are tickets?
Most requests are short: a typical ticket is about 26 words, roughly two sentences. A small number are long email threads with hundreds of words. Our models need to handle both the quick one-liners and the occasional lengthy conversation.

### `03_ticket_length_by_class.png` – Ticket length by support group
Hardware and Administrative-rights tickets tend to be longer, often because people describe troubleshooting steps they have already tried. Storage and Access requests are usually short and to the point ("please increase my folder size", "please reset my password").

### `04_top_words_overall.png` – Most frequent words
The most common words are polite phrases such as *please*, *hi*, *thanks*, and *regards*, plus days of the week. They appear in every type of request, so they tell us almost nothing about which team should handle a ticket. A good routing system has to look past them to the words that really matter.

### `05_distinctive_words_by_class.png` – Most distinctive words per support group
Each team has its own vocabulary. Purchase requests mention *PO*, *ordered*, and *receipts*; Storage requests mention *folder* and *size*; HR Support sees *new starter* and *timecard*. The number next to each bar shows how many times more likely that word is in this team's tickets than in others, so a word like *receipts* (358×) is a very strong clue. These clues are what the routing models learn to recognise automatically.

### `06_cleaning_removed.png` – What cleaning removed
Many tickets are automatic or recurring messages, such as the same purchase-order notice sent again and again with only the date changed. If identical copies ended up both in the material the system learns from and in the material we test it on, the test would be like giving students the exam answers in advance. We therefore kept just one copy of each repeated ticket (811 removed) and dropped 48 near-identical tickets that different people had sent to different teams, because we cannot tell which routing was right. In total less than 2% of the data was removed, leaving 46,978 tickets.

### `07_split_distribution.png` – Train, validation and test sets
We divided the tickets into three groups: 70% for the system to learn from (training), 15% to check its progress while it learns (validation), and 15% kept completely hidden until the very end (test). The test set acts like a final exam the models have never seen. The chart shows that each group contains the same mix of support teams, so the final exam is a fair reflection of real workload.

### Preparing new tickets (text only, no figure)
The original tickets had already been tidied by the dataset's author: lowercase letters only, no punctuation or numbers, and many filler words and personal details removed. Any new ticket typed into our demo goes through the same tidying first, so the system sees text in the form it learned from. A few everyday words were removed by the original anonymisation (for example *email* and *wifi*), so we translate them into equivalents the system knows (*mail*, *wireless*).

### Label conventions (text only, no figure)
The teams follow this organisation's own rules, which are not always obvious. For example, Outlook problems are usually handled by the Administrative-rights team, and requests for access to restricted physical areas go to HR Support. Some requests are genuinely ambiguous: tickets mentioning *password* are mostly handled by Access but sometimes by Hardware. This ambiguity limits how perfect any automatic system can be, and it is one reason we keep a human reviewer in the loop.

## Analysis / Results tab – Baseline model

### What the baseline is (text only)
Before building neural networks we built a simple, well-understood "reference" model. It scores every word in a ticket by how unusual and informative it is, then adds up the evidence for each team and picks the team with the most. It is similar to a person scanning for keywords such as *PO* or *timecard*. The neural networks must beat this reference to justify their extra complexity.

### `08_baseline_tuning.png` – Choosing the baseline's settings
We tried 24 combinations of settings and judged each on the validation tickets, never on the final test tickets. The best version looks at single words and two-word phrases (e.g. *shared folder*) and gives extra importance to mistakes on the smaller teams. Those choices raised the score from about 0.82 to 0.87.

### `09_baseline_f1_by_class.png` – How well each team is recognised
On the hidden test tickets the baseline routes **86% of tickets to the correct team**. Purchase and Storage requests are recognised best, because they use very specific words. Administrative rights is hardest: it is the smallest team, and its tickets (software installs, Outlook problems) often read like Hardware problems.

### `10_baseline_confusion_matrix.png` – Where tickets go wrong
Each row is the team that really handled a ticket, and each column is where the model sent it. The dark diagonal shows correct routing. The biggest mix-up is that 15% of Administrative-rights tickets are sent to Hardware. Most other errors happen between the large, broad teams (Hardware, HR Support, Miscellaneous), whose requests overlap.

### `11_baseline_top_words.png` – What the baseline pays attention to
These are the words that most strongly push the model towards each team. They match common sense: *monitor* and *laptop* point to Hardware, *folder* and *mailbox* to Storage, *outlook* and *upgrade* to Administrative rights. This shows the model learned sensible clues rather than accidents of the data.

### Confidence (text only)
When the baseline is right it is usually very sure (typical confidence 93%). When it is wrong it is usually unsure (typical confidence 51%). This suggests a safety rule: automatically route only the tickets the model is confident about, and send the uncertain ones to a person. We will measure this rule properly in the evaluation stage.

## Analysis / Results tab – Neural network #1: ANN

### What the ANN is (text only)
Our first neural network reads the same keyword scores as the baseline, but instead of adding up evidence in one step it passes it through a "hidden layer" of 256 small calculators. Each calculator can learn its own combination of clues, for example that *access* together with *card* means a building pass, while *access* together with *confluence* means a software permission. In principle this lets the network recognise patterns that a simple keyword tally misses.

### `12_ann_architecture.png` – How the ANN is built
A ticket flows left to right. Its words are converted into importance scores, the scores are rescaled so long and short tickets are comparable, and a layer of 256 units looks for combinations of clues. During training, half of these units are randomly switched off at each step ("dropout"), which stops the network from simply memorising tickets. The final layer turns everything into eight probabilities, one per support team.

### `13_ann_training_curves.png` – Watching the ANN learn
The blue line shows performance on tickets the network is studying. The orange line shows performance on validation tickets it has not studied. Both improve at first. After the fourth pass through the data the network keeps getting better on its study material (reaching 98%) but no better on new tickets. This is called overfitting, a sign it has started to memorise rather than learn. We automatically keep the version from the fourth pass.

### `14_ann_confusion_matrix.png` – Where the ANN goes wrong
The ANN makes the same kinds of mistakes as the baseline. The broad teams (Hardware, HR Support, Miscellaneous) get mixed up, and 14% of Administrative-rights tickets are sent to Hardware.

### Result in plain words (text only)
The ANN routes **84.7% of test tickets correctly**, slightly below the simpler baseline (86.1%), even though it has more than 12 times as many adjustable parameters. Getting there took several adjustments, most importantly rescaling the input so long tickets do not dominate; before that the ANN reached only 82.5%. The lesson: when a model can only see *which words* appear, a well-tuned simple model is already near the limit, and a bigger network mostly memorises. To do better, a model must understand *word order and context*, which is what our next two neural networks are designed to do.

## Analysis / Results tab – Neural network #2: Bidirectional LSTM

### What the BiLSTM is (text only)
Our second neural network reads a ticket the way a person does: one word at a time, in order, keeping a running memory of what it has read so far. It actually reads each ticket twice, once forwards and once backwards, so every word is understood with the context on both sides. It also learns its own "meaning map" for words, placing words used in similar situations close together. Built-in gates decide what to remember and what to forget as it reads, which helps with longer tickets.

### `15_bilstm_architecture.png` – How the BiLSTM is built
Each word is first turned into a number, then into a list of 128 numbers describing its meaning (the embedding). Two memory units then read the ticket in opposite directions, and their final summaries are combined. As with the ANN, dropout prevents memorisation, and the last layer produces eight probabilities, one per team. We compared three versions (different memory cells and summary methods) on validation tickets and kept the best.

### `16_bilstm_training_curves.png` – Watching the BiLSTM learn
The pattern matches the ANN. Performance on new (validation) tickets peaks after the third pass through the data, while performance on studied tickets keeps rising. Training was stopped automatically and the best version kept.

### `17_bilstm_confusion_matrix.png` – Where the BiLSTM goes wrong
The mistakes follow the same pattern as the other models: confusion among the broad teams, and Administrative-rights tickets sent to Hardware.

### What the word map learned (text only)
The network learned sensible word relationships on its own, without being told. *Folder* ended up next to *mailbox*, *storage* and *shared*; *outlook* next to *exchange*, *upgrade* and *windows*; *PO* next to *purchase* and *replenishment*; *timecard* next to *starter*, *leave* and *internship*. This shows it captured the vocabulary of each team.

### Result in plain words (text only)
The BiLSTM routes **83.2% of test tickets correctly**, below both the ANN (84.7%) and the baseline (86.1%). The main reason is the data itself. The dataset's authors had already removed punctuation and most small connecting words (*I, my, in, the* and so on) to anonymise the tickets, so much of the word order a sequence model relies on was gone before it saw the text. Learning word meanings from scratch with about 33,000 tickets is also hard. The next model addresses exactly this: it starts from a language model already trained on billions of words.

## Results tab – Final comparison (Stage 5)
*These figures update automatically when DistilBERT is added; the numbers below are for the baseline, ANN and BiLSTM.*

### `21_model_comparison.png` – Which model routes best?
Each row is one model, scored on 7,047 tickets none of them had seen before. The simple keyword-based baseline is the most accurate (86.1%), followed by the ANN (84.7%) and the BiLSTM (83.2%). We also tried combining models by averaging their opinions, but on the validation tickets this did not beat the baseline alone. The differences are small, only a few tickets in a hundred, which tells us all three approaches find essentially the same clues in this data.

### `22_per_class_f1_heatmap.png` – Strengths by team
Darker squares mean better recognition. Every model finds Purchase, Storage and Access easiest, because those requests use very specific words. Every model finds Administrative rights hardest. The ranking of teams is the same for all models, so the difficulty comes from the tickets themselves, not from any one method.

### `23_coverage_vs_accuracy.png` – Trading automation for accuracy
Imagine the system only routes the tickets it is confident about and hands the rest to a person. Moving right on this chart, the system routes fewer tickets by itself, but the ones it does route are more often correct. Routing everything automatically gives about 86% accuracy; routing the most confident 70% gives about 96%.

### `24_routing_outcome_per_100.png` – What happens to 100 tickets
This is the practical proposal. With a confidence rule chosen in advance on validation tickets, out of every 100 incoming tickets, **68 are routed automatically to the correct team**, **3 are routed automatically to the wrong team**, and **29 are sent to a person**. That means about seven in ten tickets need no human triage at all, and the review queue catches about **78% of all the mistakes** the system would otherwise make.

### `25_confidence_correct_vs_wrong.png` – Why the safety rule works
The blue line shows tickets the model got right, and the orange line tickets it got wrong, arranged by how confident it was. Correct answers pile up at high confidence; wrong answers sit mostly at low confidence. The dashed line is the review threshold: everything to its left goes to a person. Because the two groups are so well separated, a simple confidence cut-off filters out most errors.

### `26_top_confusions.png` – The most common mistakes
The most frequent mix-ups are among the three broad teams: Hardware, HR Support and Miscellaneous. Their requests genuinely overlap (an equipment request for a new starter, for example, could plausibly go to either Hardware or HR). Administrative-rights tickets about software and Outlook are also often mistaken for Hardware problems.

### Error analysis (text only)
About 8 in every 100 test tickets are misrouted by *every* model, and for most of those the models all agree on the same alternative team. When independent models consistently "disagree with the label" in the same way, the label itself is often the problem. Reading these tickets confirms it: several are about time cards or maternity leave yet are labelled Hardware or Access, and every model sends them to HR Support. Others are simply too vague to route, such as "having issues with the loaner". This suggests a realistic ceiling of roughly 90% for this dataset, and it is why a human-in-the-loop design matters more than squeezing out another point of accuracy.

## Demo (for the video and the Results tab)

### What the demo shows (text only)
A simple web page where anyone can type a support request in everyday language. The system tidies the text, predicts which of the eight teams should handle it, and shows how sure it is. If it is confident enough, it says **"Auto-route to [team]"**. If not, it says **"Needs human review"** and offers its best guess. For the baseline model it also highlights the words that drove the decision, so a person can see *why*.

### Suggested 2-minute demo script for the video
1. **Clear case:** pick *"Our shared folder is full, could you please increase its size?"* → **Auto-route to Storage, 100%**. The highlighted words *shared*, *folder*, *increase* and *size* show why.
2. **Another clear case:** *"Please raise a PO for two new headsets…"* → **Auto-route to Purchase**, driven by *PO* and *purchase*.
3. **Vague case:** *"Can someone help me with this, it is not working since this morning."* → **Needs human review** (54% confidence, best guess Hardware). This shows the safety rule doing its job: the system knows when it does not know.
4. **Tricky case:** *"I changed my password yesterday and now I cannot log in to the VPN or my email."* → the baseline auto-routes to Access at 76%. Switch the model to ANN and the same ticket goes to review. Different models draw the line in different places.
5. Close with the "what happens to 100 tickets" chart (`24_routing_outcome_per_100.png`).

Screenshots for the website: take them in the running app after routing examples 1 and 3, so you show one auto-route and one review.
