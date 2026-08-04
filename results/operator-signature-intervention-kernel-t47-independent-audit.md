# T47 independent audit — no-go as intervention-kernel transfer

Date: 2026-08-01  
Verdict: **NO-GO CANDIDATE; HOLD CONDITIONAL MATRIX IDENTITIES; NO RUN**

The independent audit verified the exact oracle recovery, similarity of power
traces, trace perturbation bound, complete-system automorphism ambiguity, and a
legal-margin version of the commutator order test. It rejected the transfer
claim for five reasons:

1. resettable full-rank state probes are a privileged system-identification
   oracle and may require already knowing the dynamics;
2. unrestricted coordinate changes preserve exact traces but destroy Euclidean
   finite-sample conditioning and noise margins;
3. power traces identify only spectral information, while action tuples may
   require mixed-word invariants;
4. naming action operators does not recover the state-coordinate map needed by
   a state-dependent policy; and
5. explicit system identification or a permutation/equivariance-augmented
   meta-learner can implement the same construction with the same data.

T47 also substituted deterministic full-state transition operators for T46's
stochastic hidden-hypothesis likelihood kernel. It contains no `J`, posterior,
or belief update.

The only admitted continuation is a controlled-Hankel/PSR signature defined in
raw future-test probabilities with a finite-sample and minimax interaction
comparison. If that observable correction is merely spectral PSR learning, the
operator-signature lane closes without code.
