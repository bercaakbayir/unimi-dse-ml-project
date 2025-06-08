{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "86d57e23-2f92-4590-b8f9-744e37550b0a",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "def classification_metrics(y_true, y_pred):\n",
    "    classes = np.unique(np.concatenate((y_true, y_pred)))\n",
    "    metrics = {}\n",
    "\n",
    "    total_correct = np.sum(y_true == y_pred)\n",
    "    accuracy = total_correct / len(y_true)\n",
    "    metrics['accuracy'] = accuracy\n",
    "\n",
    "    precisions = []\n",
    "    recalls = []\n",
    "    f1s = []\n",
    "\n",
    "    for cls in classes:\n",
    "        tp = np.sum((y_pred == cls) & (y_true == cls))\n",
    "        fp = np.sum((y_pred == cls) & (y_true != cls))\n",
    "        fn = np.sum((y_pred != cls) & (y_true == cls))\n",
    "\n",
    "        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0\n",
    "        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0\n",
    "        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0\n",
    "\n",
    "        precisions.append(precision)\n",
    "        recalls.append(recall)\n",
    "        f1s.append(f1)\n",
    "\n",
    "    metrics['precision'] = np.mean(precisions)\n",
    "    metrics['recall'] = np.mean(recalls)\n",
    "    metrics['f1_score'] = np.mean(f1s)\n",
    "\n",
    "    return metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "89912e46-40ff-4ff3-9187-1c07a7c8e185",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
