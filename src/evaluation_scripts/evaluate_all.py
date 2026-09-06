"""
Run the full evaluation pipeline after training: generate embeddings, then generate
both result tables. One command instead of three.

Runs, in order:
  1. generate_embeddings.py   -- results/trained_models -> results/embeds
  2. generate_nn_results.py   -- results/embeds -> results/nn_results.csv
  3. generate_linear_results.py -- results/embeds -> results/linear_results.csv

After this, plot with the eval_*.py scripts (pass --pregenerated to any of them to
plot the committed pregenerated_results/ snapshot instead of your own results/).

Usage:
    python evaluation_scripts/evaluate_all.py
"""

import generate_embeddings
import generate_nn_results
import generate_linear_results


def main():
    print('=== 1/3: Generating embeddings ===')
    generate_embeddings.main()

    print('\n=== 2/3: Generating NN preference-accuracy results ===')
    generate_nn_results.main()

    print('\n=== 3/3: Generating linear preference-alignment results ===')
    generate_linear_results.main()


if __name__ == '__main__':
    main()
