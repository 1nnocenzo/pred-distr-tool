# pred-distr-tool — guida completa e runbook (stato al 2026-10-16)

Guida per ripartire da zero: cosa fa il progetto, da dove vengono i dati, com'è fatto ogni modello, come si addestra e
si valuta, cosa è stato provato (con i numeri), cosa resta aperto, e i comandi per rifare tutto. Ogni affermazione
rimanda al file che la contiene, così si può verificare.

Indice
1. Il problema in una pagina
2. Mappa del repository
3. Ambiente software
4. Dati: circuiti, dispositivi, etichette
5. Come un circuito diventa un grafo
6. Come un dispositivo diventa un grafo
7. Il modello di fidelity (architettura completa)
8. Addestramento: loss, protocollo, ricetta in due fasi
9. Split, metriche e valutazione standard
10. Storia dei modelli di F (con i numeri)
11. ESP e durata: definizioni, teste ESP, il problema del routing
12. Anatomia del compilatore: cosa dipende davvero dalla calibrazione
13. Route stage (v24–v28), fast path, light compile
14. v29: il modello unico F + durata + ESP
15. Tempi di esecuzione
16. Limiti noti e problemi aperti
17. RUNBOOK A — rifare tutto sul cluster, nell'ordine
18. RUNBOOK B — route stage e v29 fuori dal cluster
19. Regole del cluster
20. Glossario

---

## 1. Il problema in una pagina

**Obiettivo.** Dato un circuito quantistico (OpenQASM, 2–20 qubit) e un insieme di dispositivi (QPU) con la loro
calibrazione, predire **senza compilare**, per ogni dispositivo:
- **F** = *expected fidelity* del circuito compilato (formula di MQT: prodotto di (1 − errore) su ogni operazione nativa);
- **durata** dello shot (schedule ASAP del circuito compilato);
- **ESP** = *estimated success probability* (F × decoerenza dovuta all'idle dei qubit, formula MQT).

Uso: uno scheduler sceglie il dispositivo migliore per ogni circuito di una coda, o decide dove compilare.

**Dispositivi** (`createDevice.py`): tre chip in stile IQM, gate nativi `r` (1 qubit), `cz` (2 qubit), `measure`.

| dispositivo | qubit | calibrazione |
| --- | --- | --- |
| EQE1_Top | 26 | metà superiore di EQE1 (53 qubit), JSON `data/2026-03-02T04_14_02.290607Z.json` |
| EQE1_Bottom | 27 | metà inferiore di EQE1 |
| QExa20 | 20 | CSV `data/qpu_dcdb_results.csv` |

Durate dei gate fisse e uguali sui tre chip: r 42 ns, cz 130 ns, measure 15 µs (`createDevice.py`).

**Compilatore di riferimento**: Qiskit 2.3.1, preset pass manager livello 2. Le etichette sono calcolate sul
circuito compilato. Il modello non compila mai a predizione.

**Domanda del paper** (npj Quantum Information, scadenza 2026-12-20): il predittore generalizza a famiglie di circuiti
mai viste (LOFO / LOGO) e segue i cambi di calibrazione?

---

## 2. Mappa del repository

| percorso | cosa contiene |
| --- | --- |
| `createDevice.py` | costruzione dei backend Qiskit (coupling map, errori, T1/T2, durate) |
| `data/` | `circuits.zip` (dataset QASM), calibrazioni EQE1 (JSON) e QExa20 (CSV) |
| `src/model/` | tool originale: `encoding.py` (circuito → DAG), `gnn.py` (GraphConvolutionSage), `best_params.json` (iperparametri del circuit encoder) |
| `src/qms/` | estratto del policy engine QMS (scheduler) |
| `evaluations/pipeline/` | scheduling dry-run (policy GNN vs Oracle / GT-Weighted / Round-Robin) |
| `evaluations/generalization/` (v1, `genstudy`) | dataset, split, metriche, I/O: riusati da tutte le versioni successive |
| `evaluations/generalization_v2/` (`gsv2`) | grafi dei dispositivi, target log F, sampler bilanciato, split di validazione |
| `evaluations/generalization_v3/gsv3/` | **codice dei modelli attuali**: `model.py` (tutti i tipi), `training.py` (loss), `devices.py` (grafi dei dispositivi, varianti di calibrazione, T1/T2, durate), `fast_encoding.py`, `groups.py` (LOGO), `router.py` |
| `evaluations/generalization_v3/scripts/run_experiment.py` | **entry point di addestramento** (lofo / logo / control / size) |
| `evaluations/generalization_v3/slurm/` | `train_cpu.sh`, `train_gpu.sh`, `pick_partition.sh`, `pick_gpu.sh` |
| `evaluations/generalization_vN/results/<run>[_sS]/<exp>/<split>/` | risultati: `metrics.json`, `history.json`, `model.pth` (seed 5 senza suffisso `_s`). **v9** = dati vecchi, **v10** = dati rigenerati (qaoa con misure finali): oggi si usa v10 |
| `evaluations/retrain/` | riaddestramento di tutti i modelli sui dati v10 (`plan.json` = ricette esatte di ogni run) |
| `evaluations/compile_check/` | **tutte le etichette**: layout del compilatore, conteggi, varianti di calibrazione, ESP / durata, etichette Sabre, anatomia della pipeline |
| `evaluations/calibration_shift/` | test di cambio calibrazione (`shift_k.py`, `run_shift.py`), politiche di decisione, ensemble |
| `evaluations/diagnostics_v4v5/` | diagnostica e studi oracolo (tracking, placement probe, anatomia) |
| `evaluations/esp/` | valutazioni ESP / durata (v13–v23): `esp_eval.py`, `esp_routed.py`, `esp_ttrack.py`, `esp_interval.py`, `hybrid.py` |
| `evaluations/synthetic_routing/` | circuiti sintetici per il routing (`synth_routing.py` 15k, `synth_layered.py` 6k) |
| `evaluations/route_stage/` | route stage v24–v28 (`stage1.py`, `prep.py`), **v29 `unified.py`**, `run_local.sh`, `RUNBOOK.md` |
| `evaluations/fastpath/` | percorso esatto senza compilatore per i circuiti routing-free |
| `evaluations/light_compile/` | switch + compilazione leggera (Sabre a 1 trial) |
| `evaluations/explorer/` | sorgente dell'explorer interattivo |
| `evaluations/results_report.py`, `RESULTS.md`, `TODO.md` | report automatico, risultati, findings e cose aperte |

Variabile d'ambiente necessaria per ogni script: `PYTHONPATH=src` (dalla radice del repo).

---

## 3. Ambiente software

- `environment.yml` → env conda `pred-distr`: Python 3.11, torch 2.11 (+cu128), torch-geometric 2.7, Qiskit 2.3.1,
  rustworkx, scipy, MQT bench (per le formule di fidelity).
- Le partizioni V100 del cluster non funzionano (torch 2.11+cu128 non ha kernel sm_70).
- Fuori dal cluster senza GPU: torch CPU; il codice gira su CPU per default (`--device cuda` per la GPU).
- Repository dei circuiti e dei backend sul cluster: `~/compileCircuits` (benchmark_dataset_30k, createDevice). Gli script
  ricadono su `data/` e su `createDevice.py` della radice quando `~/compileCircuits` manca.

```bash
conda env create -f environment.yml && conda activate pred-distr
unzip data/circuits.zip -d data/          # -> data/benchmark_dataset_30k/  (NB: contiene la qaoa VECCHIA senza misure finali)
export PYTHONPATH=$PWD/src
```

---

## 4. Dati: circuiti, dispositivi, etichette

### 4.1 Circuiti
- MQT Bench, 35 famiglie, 2–20 qubit. Dataset del modello: **17.056 grafi** (contro 19.465 etichette: il buco è a
  monte, in `create_pt_file_new.py`).
- Famiglie grandi: vqe_two_local 3449, vqe_real_amp 2861, vqe_su2 2858, qaoa 2706, randomcircuit 1234, qft 760, qnn 760,
  ae 758, qftentangled 746, iqpe 570; poi una coda di famiglie piccole (grover 62, hhl 44, …).
- `evaluations/generalization_v3/data/graph_dataset_expected_fidelity.pt`: lista di `Data` PyG con `x`, `edge_index`,
  `gate_qubits`, `n_qubits`, `y` (F sui 3 dispositivi), `circuit_name` = `famiglia/nome_q<N>_v<id>`.
- **qaoa rigenerata** con le misure finali (`compile_check/qaoa_measure.py`): da qui i dati "v10" e il riaddestramento
  di tutti i modelli (`retrain/`).
- Grover è escluso da tutte le ricompilazioni (QPY fino a 2,7 GB per circuito, F ≈ 0 ovunque): da dire nel paper.
- Circuiti dinamici (if_else: iqpe, dynamic_qft, …) esclusi dalle usage map e dall'ESP.

### 4.2 Dispositivi (`createDevice.py`)
- Coupling map di EQE1 (53 qubit, QB34 assente) tagliata in due metà: EQE1_Top (26) e EQE1_Bottom (27). QExa20: 20 qubit.
- Per qubit: errore `r`, errore di readout, T1, T2; per coupler: errore `cz`. I valori mancanti sono riempiti con la media.
- **Attenzione T1/T2 di QExa20**: `createDevice.py` li legge 10× troppo piccoli (ns del DCDB letti come unità 1e-4 µs).
  Le etichette ESP e il codice (`gsv3/devices.py: T_FIX`, `compile_check/esp_labels.py: T_FIX`) dividono per 10 il
  decay di QExa20. È un'ipotesi (confronto con Deng et al., arXiv:2508.21231): va verificata alla fonte.

### 4.3 Etichette (tutte in `evaluations/compile_check/`)

| etichetta | come | file |
| --- | --- | --- |
| F | Qiskit L2, MQT `expected_fidelity`; per i set ricompilati il **migliore di 3 seed** | `y` dei grafi; `v8b_orig/v8b_orig_labels.pt` (best-of-K sulla calibrazione originale) |
| layout del compilatore | layout iniziale (indici dei nodi del grafo del dispositivo), seed fisso | `compiler_layouts_v6pilot.pt` (`compile_layouts.py`) |
| conteggi nativi | r / measure / cz per circuito e dispositivo | `compiler_counts_v8.pt` (`compile_layouts.py counts`) |
| varianti di calibrazione | 100 varianti per dispositivo (jitter, shuffle, qubit cattivi, scala globale), ogni circuito ricompilato con 3 seed, best → etichetta, layout, conteggi | `v8d/` (`v8b_variants.py --n-aug 100`) |
| ESP, durata, profondità, decay | dalle compilazioni salvate (seed 0, calibrazione originale), ASAP, MQT `estimated_success_probability`, fix T di QExa20 | `esp_labels.pt` (`esp_labels.py`) |
| usage map | uso per qubit fisico / coupler | `v8u/usage_labels.pt` |
| etichette Sabre | SWAP per filo di ogni gate, schedule routed, durate (logical / wire / replay / naive / final) | `sabre_labels/` (`sabre_labels.py`) → `route_stage/results/stage1.pt` (`prep.py`) |
| sintetici | stessi tipi di etichette per 15k + 6k circuiti sintetici | `synthetic_routing/results/` |

Ogni ricompilazione è verificata (gate nativi, coupler esistenti, equivalenza). Le usage map riproducono log F a 1e-7.

**Incongruenze note delle etichette**
- F usa il migliore di 3 seed, ESP e durata usano il solo seed 0.
- `sabre_labels.py` compila il circuito DOPO `fast_encoding.to_basis` (rzz → cx·rz·cx); le etichette del dataset
  compilano il QASM originale. Stesso numero di CZ ma schedule diverso: su qaoa durata diversa in mediana del 4,6%
  (p90 15%) nel 99% dei casi (`compile_check/basis_mismatch.py`). Di conseguenza l'encoding perde l'informazione "era
  un rzz" e c'è un rumore irriducibile su qaoa / bmw_quark / randomcircuit.

---

## 5. Come un circuito diventa un grafo

`src/model/encoding.py` (lento, quello del dataset) e `gsv3/fast_encoding.py` (stesso grafo, 2,2 ms invece di 11,2):

1. barriere rimosse, transpile a livello 0 nella base OpenQASM 3 (27 gate: x, y, z, h, s, sdg, t, tdg, sx, p, rx, ry,
   rz, u, cx, cy, cz, ch, cp, crx, cry, crz, cu, swap, ccx, cswap, if_else). Gate fuori base (rzz, rxx…) vengono scomposti;
2. **nodi = istruzioni** in ordine di circuito; **archi** = istruzioni consecutive sullo stesso filo (qubit o clbit);
3. feature per nodo (40): one-hot del gate (27 + measure = 28), sin/cos di fino a 3 parametri (6), arità, numero di
   controlli, numero di parametri, flag "sul cammino critico", fan-in, fan-out;
4. `gate_qubits` (N, 3): indici dei qubit logici operandi (−1 se assente); `n_qubits`.

---

## 6. Come un dispositivo diventa un grafo

`gsv2/devices.py` + `gsv3/devices.py`:
- nodi = qubit fisici: −log(1−e_r), −log(1−e_readout), log10 T1, log10 T2, grado, + 8 autovettori del Laplaciano
  normalizzato (positional encoding: permettono di distinguere i nodi e misurare distanze), standardizzati con le
  statistiche dei 3 dispositivi originali;
- archi = coupler, feature −log(1−e_cz);
- attributi non standardizzati usati dalle teste fisiche: `phys` (P, 3) = [ε_r, ε_readout, ε_cz medio dei coupler
  incidenti], `coh` = 1/min(T1, T2) in 1/µs (con il fix di QExa20), `tg` / `tcz` durate per qubit / coupler,
  `coupler_pairs`, `coupler_eps`;
- le varianti di calibrazione passano da `variant_batch` (stessa standardizzazione dei dispositivi originali).

---

## 7. Il modello di fidelity (architettura completa)

Classe `SinkhornPlacementPredictor` in `gsv3/model.py`; il tipo dei modelli principali è
`sinkhorn_rf_route_czc` (il nome elenca le opzioni: Sinkhorn placement, flag routing-free, stima di routing, CZ vincolati).

```
circuito (DAG dei gate, 40 feature)                    dispositivo d (grafo dei qubit fisici)
   │ NodeSage (GraphConvolutionSage, hidden 142,          │ GINE × 3 sul coupling map (+ PE Laplaciano)
   │ 3 conv + 1 residual, bidirezionale)                   ▼
   ▼                                                    p_d  (P, 128) qubit fisici, d̄ (media)
 g  (G, 128) un embedding per gate (gate_proj)
   │ media/max sui qubit operandi + GINE × 2 sul grafo di interazione (archi = gate multi-qubit)
   ▼
 q  (Q, 128) un embedding per qubit logico
   │ score(q, p) = (W_q q)·(W_p p)/√128   → Sinkhorn (τ 0,3, 20 iterazioni, righe dummy per i qubit fisici liberi)
   ▼
 A  (B, D, Qm, P) placement soft: dove finisce ogni qubit logico (≈ layout iniziale del compilatore)
   │ per gate g e dispositivo d:
   │   contesto = A·p dei suoi operandi (media e range), errori ε̄ = A·phys dei suoi operandi,
   │   distanza attesa tra operandi = Aᵀ H A (H = hop sul coupling map)
   │   + flag routing-free (VF2, come VF2Layout) + stima di routing (router greedy)
   ▼
 MLP [512, 256, 128] → n_ops(g, d) = (n_r, n_measure, n_cz) ≥ 0      (conteggi nativi per gate)
   │ _czc: n_cz = base(tipo di gate) · (1 + κ · hop extra · (1 − routing_free))
   │       → un circuito senza routing ha esattamente il conteggio della scomposizione
   ▼
 log F(c, d) = − Σ_g Σ_k s_k · n_k(g, d) · ε̄_k(g, d)        k ∈ {r, readout, cz},  s_k = scale apprese (init 1)
```

Perché è fatto così:
- **log F è una somma di termini ≥ 0** → F ≤ 1 per costruzione, e rispecchia la forma prodotto della fidelity attesa;
- **il modello impara la compilazione** (conteggi e placement); **le grandezze degli errori vengono dal dispositivo**:
  se la calibrazione cambia, la predizione cambia senza riaddestrare (è ciò che dà il "tracking");
- il placement è un layer differenziabile (Sinkhorn = rilassamento di un assegnamento uno-a-uno);
- `hard_placement` (Hungarian sulla matrice di Sinkhorn) dà un qubit fisico per qubit logico; serve alle teste ESP.

Varianti nel codice (stesso file, attivate dal nome): `_ep` / `_ep3` (prior di errore nel placement), `_mlps` (scorer
MLP), `_vf2` (prior dello score di VF2Layout), `_usage` (v11, usage map), `_pgnn_usage` (v12, rete sul circuito
piazzato), suffissi ESP `_esp`, `_espa`, `_espi`, `_esps`, `_espf`, `_espfi`, `_espft`, `_espfr`, `_espp` (sez. 11).

---

## 8. Addestramento: loss, protocollo, ricetta in due fasi

Codice: `gsv3/training.py` (`train_model`), entry point `scripts/run_experiment.py`.

**Protocollo**: Adam, lr 3e-4 (fine-tuning 1e-4), gradient clipping 1,0, batch 32, sampler bilanciato per famiglia
(`--balance-alpha`, 0 nei modelli attuali), early stopping sulla MSE lineare di validazione (patience; `--keep-last`
nel fine-tuning), validazione `--val-mode random` per i modelli attuali.

**Loss** (pesi dei modelli attuali):

| loss | flag | cosa |
| --- | --- | --- |
| mixed | `--loss mixed` | MSE su F + 0,1 · MSE su log max(F, 1e-3) |
| layout (v7) | `--layout-lambda 0.1 --layout-loss region_dist --layouts-path …` | quali qubit fisici usa il compilatore e quanto lontani mette i qubit che interagiscono (non il qubit esatto) |
| conteggi (v8) | `--count-lambda 0.1 --count-weights 0.1,0.1,1 --counts-path …` | conteggi nativi r / measure / cz predetti vs veri, peso alto sui CZ |
| change loss | `--calib-aug v8d --orig-best … --delta-lambda 1` | ogni batch su calibrazione originale e su una variante casuale: Δ log F predetto vs vero |
| correlazione | `--delta-corr-lambda 0.1` | 1 − Pearson(Δ predetto, Δ vero) per batch e dispositivo (v8kdc10) |
| aux sulle varianti | `--aug-aux` | layout e conteggi anche sulla variante (v8kdl40) |
| ESP | `--esp-path esp_labels.pt --esp-lambda 0.1` | decay / durata / profondità (teste ESP) |

**Ricetta in due fasi** (ricette esatte in `retrain/plan.json`):
1. **v8k da zero**: `sinkhorn_rf_route_czc` + mixed + layout + conteggi;
2. **fine-tuning** da v8k, 40 epoche, lr 1e-4, change loss sulle 100 varianti v8d: v8kd40 → + `--aug-aux` (v8kdl40) →
   + `--delta-corr-lambda 0.1` (**v8kdc10**, modello principale di F). Con `_espi` / `_espfi` e `--esp-path` si
   ottengono v8kdi10 / v8kdfi10 (F + ESP).

---

## 9. Split, metriche e valutazione standard

**Split standard** (seed 5, 6, 7):
- `lofo qaoa`, `lofo qnn`: leave-one-family-out (la famiglia di test non è mai vista);
- `logo variational`: leave-one-group-out, toglie vqe_* + qnn insieme (`gsv3/groups.py`), con il sottoinsieme qnn;
- `control x`: split casuale stratificato per F (`random_seed5/6/7`), 30% test; riferimento in-distribution.

**Metriche**: R² di F (media sui 3 dispositivi), regret della scelta del dispositivo (F perduta seguendo il modello),
quota di dispositivo giusto.

**Test di calibrazione** (`calibration_shift/shift_k.py`, `shift_k.sh predict_report`): 457 circuiti mai visti, 39
calibrazioni (3 chip × originale + 12 varianti: ×0,5, ×2, shuffle, jitter, ±20–30%…), verità = migliore di 5 seed.
**Tracking** = correlazione (non R²) tra Δlog F predetto e vero sotto un cambio di calibrazione; "locale" = shuffle /
jitter, il caso difficile. Soffitto di rumore del compilatore ~0,8–0,9 (split-half).

**Valutazione standard di un candidato** (CLAUDE.md): (1) 4 split × 3 seed; (2) test di calibrazione; (3)
`tracking_anatomy.sh` (dove sbaglia e perché); (4) `placement_probe.sh` e studi oracolo; (5) `decision_policies.py`
(regola top-k adattiva: compila sui dispositivi predetti entro δ dal migliore; cold start su flotte 3–33 backend,
aggiornamenti di calibrazione); (6) ensemble dei seed come opzione (3× il costo di inferenza).

---

## 10. Storia dei modelli di F (con i numeri)

| versione | cosa aggiunge | risultato chiave |
| --- | --- | --- |
| v1 | GNN solo circuito, 3 uscite fisse | R² 0,996 in-distribution, ma perde l'ordine dei dispositivi sulle famiglie nuove |
| v2 | target log F, encoder del dispositivo (GINE), sampler bilanciato | training instabile (esplosioni) |
| v3 | clipping, lr 3e-4, selezione su MSE lineare; `xattn` (cross-attention qubit logici → fisici) | stabile |
| v4 `phys` | testa fisica: conteggi appresi × errori calibrati | qaoa 0,88 / qnn 0,67 / variational 0,90; tracking ≈ 0 |
| v5 | placement Sinkhorn | instabile tra seed |
| v6 / v7 | loss sul layout del compilatore (v7: regione + distanza) | tracking locale 0,43 |
| v8a | + loss sui conteggi | qaoa 0,892 / qnn 0,843 / variational 0,880 / control 0,996; tracking 0,46 |
| v8r / v8s | flag routing-free, stima di routing + loss pesata sui CZ | |
| v8k | CZ vincolati alla scomposizione quando non serve routing | base di tutti i successivi |
| v8kd40 | + fine-tuning con change loss su 100 calibrazioni | qaoa 0,965 / qnn 0,855 / var 0,918; tracking 0,60 |
| v8kdl40 | + layout e conteggi sulle varianti | tracking 0,63; regret dopo un cambio −43% ripredicendo |
| **v8kdc10** (principale) | + loss di correlazione 0,1 | qaoa 0,970 ± 0,023, qnn 0,847 ± 0,123, var 0,939 ± 0,018, control 0,996; tracking locale 0,66 ± 0,02 (ensemble 0,69), globale 0,95–0,98 |

Provati e scartati: prior di errore per tipo (ep3), scorer MLP, lr del placement ×10, griglia τ / peso / epoche,
prior VF2 (v8kdv40), v11 usage map (tracking 0,59), v12 rete sul circuito piazzato (0,857 col layout vero del
compilatore, nessun guadagno col placement appreso), loss di reazione, v9 (feature relative).

**Perché il tracking si ferma a ~0,6–0,66** (`diagnostics_v4v5/results/report_tracking_*`): gli embedding contengono
tutti gli errori, ma il placement si muove la metà del compilatore (accordo di reazione 0,52 vs 0,87 tra seed del
compilatore) e pesa troppo i CZ rispetto al readout; peggio sui chip EQE1 e sui circuiti di 1–5 qubit. Lo studio oracolo:
non sono epoche né conteggi; la testa (layout iniziale + totali) non può descrivere il routing. Più calibrazioni in
training aiutano (8 → 100 calibrazioni: tracking 0,45 → 0,59).

---

## 11. ESP e durata: definizioni, teste ESP, il problema del routing

**Definizioni** (MQT, `compile_check/esp_labels.py`):
- durata = schedule ASAP del circuito compilato con r 42 ns, cz 130 ns, measure 15 µs;
- per ogni qubit attivo q: idle_q = durata − busy_q;
- **decay** = log ESP − log F = − Σ_q idle_q / min(T1, T2)_q;  **ESP = F · e^decay**.

**Ordini di grandezza importanti**
- la misura (15 µs) domina la durata (mediana ~20 µs); T ~15–40 µs → l'idle pesa molto;
- circuiti **routed**: log ESP mediano −17 (ESP ≈ 3e-8). Il 71% dei casi routed ha ESP < 1e-3;
- **sensibilità**: Δlog ESP ≈ (Δdurata / durata) · |log ESP|. Un 5% di errore sulla durata sposta log ESP di ~1
  (ESP sbagliato di ×2,7). Per questo gli errori ESP si danno in log e solo dove ESP > 1e-3;
- attenzione alle unità: "abs Δlog ESP 0,7" significa un fattore e^0,7 ≈ 2, **non** 0,7%.

**Teste ESP sul modello di F** (`gsv3/model.py: esp_features`, `esp_from_features`):

| versione | suffisso | idea | note |
| --- | --- | --- | --- |
| v13 | `_esp` | durata = qubit più occupato × fattore appreso; MLP sugli embedding | |
| v14 | `_espa` | analitica: solo struttura, niente embedding | generalizza meglio |
| v16 (v8kdi10) | `_espi` | idle totale predetto direttamente × media 1/T; misure esatte | |
| v17 (v8kdg10) | `_esps` | schedule ASAP del DAG logico come base, MLP corregge | |
| v18 | — | tempi per qubit / coupler, varianti di T1/T2 e durate | |
| v19 / v19i (**v8kdfi10**) | `_espf` / `_espfi` | testa a formula: MLP legge lo schedule a tempi di riferimento, la formula usa i tempi reali; decay per qubit col placement hard | R² ESP 0,961 / 0,937 / 0,911 / 0,986; R² F 0,964 / 0,904 / 0,954 / 0,994 |
| v20 | `_espp` | placement ESP proprio, addestrato sul qubit esatto | |
| v21t | `_espft` | tabella di scomposizione fissa per i routing-free | durata routing-free 0,5–1,5% |
| v22 | — | + 15k circuiti sintetici di routing | |
| v23 (v8kdfy10) | `_espfr` | feature di un router approssimato | |

**Il problema**: i circuiti routing-free sono risolti (durata 0,5–1,5%); i **routed** su famiglie nuove no: durata
v19i 15% (qaoa), 15,5% (variational), sottostimata; errore ESP ×1,7–1,9 (`esp/results/report_esp_routed_v10_v19i.md`);
un limite superiore al 90% calibrato sulle famiglie viste copre solo il 35–53%. Pavimento del compilatore (seed 1 vs
seed 0): 5,1% qaoa, 2,7% variational, 4,6% control.

---

## 12. Anatomia del compilatore: cosa dipende davvero dalla calibrazione

`compile_check/pipeline_anatomy.py`, report in `compile_check/results/pipeline_anatomy*/report.md` (200 circuiti × 4
varianti locali × 3 seed):
- Qiskit L2: layout Sabre, routing e ottimizzazione **ignorano gli errori**. Solo **VF2Layout** (routing-free) e
  **VF2PostLayout** (routed) scelgono DOVE va la struttura, con uno score di errore medio;
- conteggio CZ e stato routing-free non cambiano mai con la calibrazione (100%); stessa struttura compilata nel 93%
  (routing-free) / 79% (routed) dei casi;
- rieseguire solo lo step VF2 sul circuito che valuta riproduce il Δlog F della ricompilazione con corr 1,000 (esatto 88%);
- tempi: routing-free 2,3 ms (≈ lo step VF2), routed 40 ms contro 8 ms dello step VF2.

Conseguenze: (a) i routing-free si possono calcolare **esattamente** senza compilatore; (b) la struttura routed si
può calcolare **una volta per topologia** e poi rifare solo lo score per calibrazione; (c) il tracking del modello è
limitato dal suo placement appreso, mentre quello del compilatore è uno score semplice.

---

## 13. Route stage (v24–v28), fast path, light compile

### 13.1 Route stage (`route_stage/stage1.py`)
Rete separata che impara **la Sabre** (etichette solo per l'addestramento, mai input):
- input per gate: feature del nodo, feature del filo (`wire_feats`), feature del router greedy per dispositivo (SWAP
  per slot e hop, `prep.prior_route`);
- rete: MLP di ingresso + 6 layer di message passing (predecessori, successori, globale, LayerNorm, hidden 96);
- uscite per gate e dispositivo: tasso di SWAP per ogni filo operando (Poisson), attesa extra del gate (Huber);
  stage 2 = log(durata finale / schedule routed) per circuito (traduzione + ottimizzazione);
- durata = schedule ASAP dei gate con quegli SWAP e attese × e^stage2; in valutazione la mediana di 24 campioni
  Poisson (lo schedule delle medie sottostima: disuguaglianza di Jensen).

| versione | cambiamento | durata routed (mediana, vs Sabre) |
| --- | --- | --- |
| v24b | base, 40 epoche, 3 seed | qaoa 13,7%, variational 5,4%, control 5,3%, LOSO 13,8% |
| v25 | loss sulla durata attraverso uno schedule differenziabile (`schedule_torch`) | LOSO 7,6%, control 3,3%, qaoa 13,9% |
| v26 | + contesa del router greedy | = v25, scartato |
| **v27** (adottato) | + 6k circuiti sintetici della classe densa, irregolare, poco parallela (`synth_layered.py`) | **qaoa 8,8%** (vs dataset 6,4%), variational 5,4%, control 3,2%, LOSO 7,4%; classe tenuta fuori 11,8% |
| v28 | Sabre leggera (Qiskit, 1 trial) come input al posto del router greedy | **sospeso**: i dati `sabre_light.pt` non sono mai stati prodotti |

Diagnosi chiave (v24, `results/report_analysis_v24.md`): l'errore non veniva dai conteggi di SWAP ma dalle **attese**
sul cammino critico (vere 18,7% della durata, predette 5,4%), corrette da v25 + v27.

### 13.2 Fast path (`fastpath/`), circuiti routing-free
Una volta per circuito e topologia: switch VF2, una compilazione di riferimento, enumerazione degli embedding. Per
calibrazione: score di VF2Layout su ogni embedding, il migliore, poi F ed ESP con le formule MQT. Risultato: switch con
precisione = recall = 1,000; errore su log F mediano 1e-3; tracking 0,94 (modello 0,66); 0,9 ms per calibrazione contro
2,4 ms ricompilando. Limite: i pareggi nello score (68% dei casi), che Qiskit risolve con l'ordine degli embedding.

### 13.3 Light compile (`light_compile/`)
Switch (controlli economici + VF2) → routing-free: L2 esatto; routed: L2 con la Sabre leggera (1/1/1 trial invece di
20/2/20), fisica calcolata esattamente. Su 800 circuiti × 3 dispositivi:

| | durata routed | abs Δlog ESP (ESP > 1e-3) | abs Δlog F |
| --- | --- | --- | --- |
| light compile | 5,4% (qaoa 6,6%) | 0,29 | 0,50 |
| pavimento (seed 1) | 3,9% (qaoa 4,8%) | 0,20 | 0,37 |

Tempo: −70% sul totale; 0,15–0,27× la compilazione completa sopra 50 gate 2q.

---

## 14. v29: il modello unico F + durata + ESP

File `route_stage/unified.py`, test `route_stage/test_unified.py`, lancio `run_local.sh test29 | smoke29 | train29`.

**Perché.** Nel percorso v28 c'erano due reti e due placement: F da v8kdc10 (placement appreso), decay dai qubit
fisici della Sabre leggera; e F costava un secondo forward (10–40 ms).

**Architettura.**
```
circuito ──► BACKBONE = rete di F di v8kdfi10 (congelata; testa ESP propria spenta: skip_esp)
              │  log F (B, D)            ← F, placement e tracking identici a v8kdfi10
              │  g (G,128) embedding dei gate, dist (G, D) distanze attese tra operandi, A placement
              ▼
           ROUTE STAGE v27 (Stage1) con input extra: Linear(g)→32, log1p(dist)
              │  tassi di SWAP per filo, attese, stage 2
              │  routing-free (VF2 del backbone, niente compilatore) → zero SWAP e zero attese
              ▼
           durata = schedule_torch(SWAP, attese) × e^stage2      (MC: 24 campioni in un solo passaggio torch)
              │
           busy_q = Σ (tempo dei gate + tempo SWAP sul filo q);  idle_q = durata − busy_q
           T del qubit fisico scelto dal placement hard del BACKBONE (lo stesso di F)
              ▼
           decay = −[ e^a · Σ_q idle_q / min(T1,T2)_q  +  softplus(b) · durata · media 1/min(T1,T2) ]
           log ESP = log F + decay          (a, b da una piccola testa, inizializzata a a = 0, b piccolo)
```
- Loss: quelle di v27 (SWAP Poisson, attese Huber, stage 2, durata) + Huber su log(−decay) contro `esp_labels.pt`.
  `--esp-grad head` (default): la loss ESP addestra solo la testa ESP; `route`: anche SWAP / attese / stage 2.
- Il checkpoint salvato non contiene i pesi del backbone; si ricaricano da v8kdfi10.
- La valutazione dà: durata (vs Sabre e vs dataset, per fascia di gate 2q), R² F, R² ESP, abs Δlog ESP (tutti, routed,
  routing-free), scelta del dispositivo per ESP, accordo dello switch, e i **tempi della predizione completa per
  circuito** (router greedy + forward + MC, batch 1, 1 thread) accanto alla light compile.

**Criteri di adozione** (RUNBOOK): durata routed non peggiore di v27; abs Δlog ESP routed sotto v8kdfi10 (0,53 / 0,65 /
0,30) e verso la light compile (0,29); tempo totale sotto la light compile nella stessa fascia.

**Stato**: scritto, pushato, mai eseguito. Il primo passo è `test_unified.py`.

---

## 15. Tempi di esecuzione

| cosa | tempo | fonte |
| --- | --- | --- |
| encoding veloce | 2,2 ms mediana | `timing_v8/` |
| modello di F, 3 dispositivi (encoding + forward, 1 thread) | 12,9 ms; ~1,2 ms per dispositivo in più | `timing_v8/timing_scaling_*.log` |
| compilazione completa + fidelity | ~21 ms per dispositivo; q16–20 a 27 dispositivi 2,96 s | idem |
| compilazione L2 per circuito (3 dispositivi) per gate 2q: 0–20 / 20–50 / 50–200 / 200–600 / 600+ | 33 / 86 / 531 / 1845 / 2786 ms | `light_compile/results/report.md` |
| light compile (solo routed) stesse fasce | 19 / 37 / 69 / 132 / 190 ms | idem |
| fast path per calibrazione (routing-free) | 0,9 ms (ricompilare 2,4 ms) | `fastpath/results/report_validate.md` |
| Sabre leggera per (circuito, dispositivo) | ~6–10 ms (randomcircuit ~40) | `route_stage/RUNBOOK.md` |
| schedule MC vecchio (loop Python) | 2 / 9 / 20 ms per dispositivo con 100 / 500 / 1500 gate (misura sintetica) | questa sessione |
| percorso v29 completo | **da misurare** (lo fa `unified.py`) | |

---

## 16. Limiti noti e problemi aperti

1. **ESP routed esponenzialmente sensibile alla durata** (sez. 11): l'accuratezza ESP è limitata da quella della durata
   × |log ESP| ≈ 17.
2. **Pavimento del compilatore**: nessun predittore della "durata del seed 0" scende sotto 3,9% (qaoa 4,8%);
   abs Δlog ESP 0,20. La Sabre è casuale.
3. **qaoa e la scomposizione di rzz**: rumore irriducibile del 4,6% finché l'encoding non ha i tipi rzz / rxx.
4. **Generalizzazione per classi**: v27 migliora qaoa solo perché la sua classe è nei sintetici; classe tenuta fuori 11,8%.
5. **Placement appreso**: qubit esatto del compilatore 7–16%, regione 75–93% → tracking 0,66; il fast path (score VF2)
   arriva a 0,94 sui routing-free.
6. **T1/T2 di QExa20**: correzione ×10 da verificare.
7. **Etichette miste**: F best-of-3, ESP/durata seed 0.
8. **Riuso del test**: ~30 versioni scelte sugli stessi split. Prima del paper: congelare il modello e valutarlo una
   volta su un test mai usato (famiglia tenuta fuori o circuiti nuovi), con lo spread dei seed (qnn ±0,05–0,08).
9. **Più dispositivi e topologie**: tutto è su 3 chip con durate uniformi; leave-one-device-out da fare.
10. **Domanda aperta per i routed**: il modello vale la candela contro la light compile? Lo decide la tabella dei tempi
    e degli errori di v29.

---

## 17. RUNBOOK A — rifare tutto sul cluster, nell'ordine

Tutti i comandi dalla radice del repo, env `pred-distr` attivo, `PYTHONPATH=src`. Prima di ogni lancio confrontare code e
tempi di partenza su tutte le partizioni (sez. 19).

### A1. Dispositivi e dataset
Il dataset originale (`graph_dataset_expected_fidelity.pt`: grafi + F sui 3 dispositivi) viene da
`create_pt_file_new.py` nel repo `~/compileCircuits` (fuori da questo repository): QASM → `qasm3.load` → transpile
livello 0 → `create_dag`, etichette = compilazione L2 + MQT expected fidelity.
```bash
python createDevice.py                                    # carica i backend e stampa il riepilogo; verificare T1/T2 di QExa20 (sez. 4.2)
python evaluations/generalization_v3/scripts/build_qubit_dataset.py   # aggiunge gate_qubits / n_qubits -> generalization_v3/data/
python evaluations/timing_v8/check_fast_encoding.py        # fast_encoding == grafi del dataset
```

### A2. Etichette (`evaluations/compile_check/`, job SLURM in loop, niente array)
Ordine (ogni script ha `run --task-id i --n-tasks N` e poi `merge`; i `.sh` accanto sono i launcher):
1. `compile_layouts.py run … --exclude grover --save-qpy` → `merge` → `counts` (layout, QPY, conteggi);
2. `qaoa_measure.py` (qaoa con le misure finali) se si rigenera qaoa;
3. `v8b_variants.py run --k 3 --n-aug 100 --out results/v8d` → `raw` → `v8d_merge.sh` (100 varianti) e
   `v8b_merge_orig.sh` (best-of-3 sull'originale → `v8b_orig/v8b_orig_labels.pt`);
4. `esp_labels.py run` → `merge` (→ `esp_labels.pt`, fix T di QExa20);
5. `sabre_labels.py run --task-id i --n-tasks 48 --out results/sabre_labels`;
6. `../synthetic_routing/synth_routing.py run … --n-circuits 15000` → `merge`; `synth_layered.py gen/sabre/prep` (via
   `run_local.sh synth2`);
7. `../route_stage/prep.py --out results/stage1.pt`.

### A3. Modello di F (split standard × seed 5, 6, 7)
```bash
LP=evaluations/compile_check/results/compiler_layouts_v6pilot.pt
CP=evaluations/compile_check/results/compiler_counts_v8.pt
BASE="--loss mixed --val-mode random --layout-lambda 0.1 --layout-loss region_dist --layouts-path $LP \
      --count-lambda 0.1 --count-weights 0.1,0.1,1 --counts-path $CP"
R=evaluations/generalization_v10/results
# fase 1: v8k da zero (uno split per job; control: FAMILIES=x)
RUN=v8k MODEL=sinkhorn_rf_route_czc ALPHA=0 SEED=5 EXPERIMENTS=lofo FAMILIES=qaoa SPLIT_GROUPS=qaoa \
  RESULTS_ROOT=$R EXTRA_ARGS="$BASE" sbatch -p cpu_sapphire --time=08:00:00 --mem=6G --export=ALL \
  evaluations/generalization_v3/slurm/train_cpu.sh
# fase 2: fine-tuning v8kdc10 (afterok sul job di v8k; --mem=8G)
FT="--calib-aug evaluations/compile_check/results/v8d --orig-best evaluations/compile_check/results/v8b_orig/v8b_orig_labels.pt \
    --delta-lambda 1 --keep-last --epochs 40 --patience 100 --lr 1e-4 --aug-aux --delta-corr-lambda 0.1 \
    --init-from $R/v8k/leave_one_family_out/qaoa/model.pth"
RUN=v8kdc10 MODEL=sinkhorn_rf_route_czc … EXTRA_ARGS="$BASE $FT" sbatch --dependency=afterok:<job v8k> …
# variante F + ESP: v8kdi10 = MODEL=sinkhorn_rf_route_czc_espi + "--esp-path evaluations/compile_check/results/esp_labels.pt --esp-lambda 0.1"
#                   v8kdfi10 = MODEL=sinkhorn_rf_route_czc_espfi, --epochs 20, --init-from il v8kdi10 dello stesso split
```
Le ricette esatte di ogni run (anche i seed 6, 7 e gli altri split) sono in `evaluations/retrain/plan.json`;
`retrain/retrain_all.py submit` le rilancia tutte con le dipendenze.

Uno split per job: `EXPERIMENTS=lofo FAMILIES=qnn SPLIT_GROUPS=qnn`; `EXPERIMENTS=logo FAMILIES=variational
SPLIT_GROUPS=variational`; `EXPERIMENTS=control FAMILIES=x`. Un job ripreso con lo stesso RUN riparte da `checkpoint.pt`.

### A4. Valutazione (dopo il control, con `--dependency`)
```bash
sbatch --dependency=afterany:<control> --cpus-per-task=8 --mem=12G --time=01:00:00 \
  evaluations/calibration_shift/shift_k.sh predict_report           # test di calibrazione -> results_k/report.md
evaluations/diagnostics_v4v5/tracking_anatomy.sh --compare <main>,<cand> --main <cand>:<kind>
evaluations/diagnostics_v4v5/placement_probe.sh
python evaluations/calibration_shift/decision_policies.py --models <main>,<cand>
python evaluations/calibration_shift/ensemble_tracking.py --models <main>
python evaluations/esp/esp_eval.py --run v8kdfi10 --kind sinkhorn_rf_route_czc_espfi --root evaluations/generalization_v10/results
GS_RESULTS_ROOT=generalization_v10 python evaluations/esp/esp_routed.py --runs v8kdfi10
python evaluations/results_report.py                                 # RESULTS.md + figures/
```
Un nuovo modello richiede: il tipo in `gsv3/model.py` (`MODEL_KINDS`, `build_model`) e nelle liste di
`run_experiment.py`, una riga in `calibration_shift/run_shift.py` MODELS e in `results_report.py` MODELS.

### A5. Fast path e light compile
```bash
bash evaluations/fastpath/launch.sh             # SLURM; senza SLURM: fastpath/run_local.sh
bash evaluations/light_compile/run_local.sh     # -> light_compile/results/report.md
```

---

## 18. RUNBOOK B — route stage e v29 fuori dal cluster

1. Codice: `git clone` / `git pull origin esp-depth`.
2. Dati (2,1 GB, `~/v24_data_bundle.tar` sul cluster con il sha256; lista in `route_stage/bundle_files.txt`):
   `stage1.pt`, grafi del dataset e sintetici, `esp_labels.pt`, liste di split v8kdfi10. Per v27/v29 servono anche
   `stage1_synth2.pt` e `synth2_graphs.pt` (`run_local.sh synth2`, già fatto).
3. Per v29: i 9 `model.pth` di v8kdfi10 dal cluster, stessi percorsi:
   `evaluations/generalization_v10/results/v8kdfi10{,_s6,_s7}/{leave_one_family_out/qaoa,leave_one_group_out/variational,random_control/random_seed{5|6|7}}/model.pth`.
4. Comandi:
```bash
cd evaluations/route_stage
PYTHONPATH=../../src python test_stage1.py                 # 10 s
PYTHONPATH=../../src python test_unified.py                # 30 s: deve finire con "all checks passed"
bash run_local.sh smoke29                                  # ~10 min: guardare durata delle epoche e loss
DEVICE=cuda GPUS="0 1" JOBS=4 bash run_local.sh train29    # 3 seed × (qaoa, variational, control, loso:dense,complete, loso:layered,seqdense)
bash run_local.sh report                                   # results/report_v24.md, sezione "v29"
```
Variabili: `THREADS`, `JOBS`, `DEVICE`, `GPUS`, `EPOCHS` (40), `SEEDS` ("5 6 7"), `MC` (24), `ESP_LAMBDA`, `ESP_GRAD`,
`C2_TARGET`, `TIME_N`. Log in `route_stage/local_logs/`. `train29` salta i run che hanno già il file di valutazione.
Dettagli e tabelle di adozione di ogni versione in `route_stage/RUNBOOK.md`.

---

## 19. Regole del cluster (PoliTo HPC)

- Mai dati o training sul login node: `srun` / `sbatch` (anche gli smoke test). `/tmp` è locale al nodo: script dei
  nodi di calcolo sotto `$HOME`.
- Prima di ogni lancio: `sbatch --test-only` su gpu_h200, gpu_a100, gpu_a40, gpu_a40_ext, cpu_sapphire,
  cpu_sapphire_ext, cpu_skylake, cpu_skylake_ext e scegliere la più veloce (`slurm/pick_partition.sh <time> <mem>`).
  gpu_h200 (compute-7-12) con `train_gpu.sh`, GPU 0000:19:00.0 rotta (`pick_gpu.sh` la salta). Niente V100.
- Nodi lenti (~30×: compute-5-6, compute-5-8): controllare le prime epoche, eventualmente `-w <nodo libero>`.
- Memoria vicina al picco: training CPU 6G, fine-tuning 8G, compilazioni 2G, test di calibrazione 12G; misurare con
  `sstat -j <id>.batch -o MaxRSS` (`sacct` / `seff` non funzionano). Niente job array; non si può alzare il tempo di un job.
- Catene con `--dependency` (`afterok` per il fine-tuning, `afterany` per i report).

---

## 20. Glossario

| termine | significato |
| --- | --- |
| F | expected fidelity MQT: Π(1 − e) sulle operazioni native del circuito compilato |
| ESP, decay | F · e^decay, decay = − Σ idle_q / min(T1,T2)_q |
| routing-free | il grafo di interazione entra nel coupling map: nessuno SWAP (VF2Layout trova un layout) |
| routed | servono SWAP (Sabre) |
| placement | dove il modello mette ogni qubit logico sul chip (Sinkhorn soft, Hungarian hard) |
| tracking | correlazione tra Δlog F predetto e vero sotto un cambio di calibrazione |
| regret | fidelity persa scegliendo il dispositivo del modello invece del migliore |
| LOFO / LOGO / LOSO | leave-one-family / group / structure out |
| change loss | loss su Δ log F tra calibrazione originale e una variante |
| v8d | le 100 varianti di calibrazione per dispositivo usate in training |
| stage 1 / stage 2 | route stage: SWAP e attese per gate / correzione per traduzione e ottimizzazione |
| light Sabre / light compile | Sabre con 1 trial / L2 con quella Sabre e fisica esatta |
| fast path | calcolo esatto dei routing-free con lo score di VF2Layout sugli embedding in cache |
| pavimento del compilatore | differenza tra due seed del compilatore: limite di qualunque predittore del seed 0 |
