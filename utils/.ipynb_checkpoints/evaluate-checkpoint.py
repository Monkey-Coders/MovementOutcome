import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
    

def evaluate(preds, labels, video_ids, aggregate_binary=False, aggregate_binary_threshold=0.5, median_aggregation=True, prediction_threshold=0.5, threshold_metrics=True, auto_threshold=False, subject=True, normalized=False):

    # Compute softmax values
    if normalized:
        softmax = preds
    else:
        softmax = np.zeros(preds.shape)
        for i in range(preds.shape[0]):
            softmax[i,:] = np.exp(preds[i,:]) / np.sum(np.exp(preds[i,:]), axis=0)
                        
    # Extract CP scores of sequence parts
    parts_scores = softmax[:,1]
    
    # Sort predictions on video ID
    video_ids_scores = {}
    video_ids_label = {}
    for i in range(parts_scores.shape[0]):
        video_id = video_ids[i]
        part_score = parts_scores[i]
        label = labels[i]
        try:
            video_ids_scores[video_id].append(part_score)
        except:
            video_ids_label[video_id] = label
            video_ids_scores[video_id] = [part_score]
        
    # Get accumulative scores
    accumulative_scores = []
    cp_labels = []
    sample_weights = []
    for video_id in video_ids_label.keys():
        label = video_ids_label[video_id]
        part_scores = np.asarray(video_ids_scores[video_id])
        if aggregate_binary:
            scores = (part_scores >= aggregate_binary_threshold).astype(int)
        else:
            scores = part_scores
        if subject:
            if median_aggregation:
                accumulative_score = np.median(scores)
            else:
                accumulative_score = np.mean(scores)
            if accumulative_score >= 0.0 and accumulative_score <= 1.0:
                accumulative_scores.append(accumulative_score)
            elif accumulative_score < 0.0:
                accumulative_scores.append(0.0)
            elif accumulative_score > 1.0:
                accumulative_scores.append(1.0)
            else:
                accumulative_scores.append(prediction_threshold)
            cp_labels.append(label)
        else:
            for score in scores:
                if score >= 0.0 and score <= 1.0:
                    accumulative_scores.append(score)
                elif score < 0.0:
                    accumulative_scores.append(0.0)
                elif score > 1.0:
                    accumulative_scores.append(1.0)
                else:
                    accumulative_scores.append(prediction_threshold)
            cp_labels += [label for i in range(len(scores))]
            sample_weights += [(1/len(scores))/len(video_ids_label.keys()) for i in range(len(scores))]
           
    # Area Under ROC Curve
    if not np.any(np.isnan(accumulative_scores)):
        if subject:
            area_under_curve = roc_auc_score(cp_labels, accumulative_scores, average='weighted')
        else:
            area_under_curve = roc_auc_score(cp_labels, accumulative_scores, average='weighted', sample_weight=sample_weights)
    else:
        area_under_curve = 0.5
            
    # Compute threshold-based evaluation metrics
    if threshold_metrics:
    
        # Aggregate scores
        if auto_threshold:
            if subject:
                fpr, tpr, thresholds = roc_curve(cp_labels, accumulative_scores)
            else:
                fpr, tpr, thresholds = roc_curve(cp_labels, accumulative_scores, sample_weight=sample_weights)
            best_sens_spec = 0.0
            best_threshold = None
            for inv_spec, sens, thres in zip(fpr, tpr, thresholds):
                spec = 1 - inv_spec
                if best_sens_spec < (sens + spec):
                    best_sens_spec = sens + spec
                    best_threshold = thres
            prediction_threshold = best_threshold
        cp_predictions = (np.asarray(accumulative_scores) >= prediction_threshold).astype(int)
        cp_labels = np.asarray(cp_labels)

        # Compute true/false positives and true/false negatives

        ## True positive (predict CP, true label is CP)
        tp = np.sum(np.logical_and(cp_predictions == 1, cp_labels == 1))

        ## True negative (predict not CP, true label is not CP)
        tn = np.sum(np.logical_and(cp_predictions == 0, cp_labels == 0))

        ## False positive (predict CP, true label is not CP)
        fp = np.sum(np.logical_and(cp_predictions == 1, cp_labels == 0))

        ## False negative (predict not CP, true label is CP)
        fn = np.sum(np.logical_and(cp_predictions == 0, cp_labels == 1))

        ## Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        ## F1-score
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = (2 * precision * recall) / (precision + recall)

        ## Sensitivity
        sensitivity = recall

        ## Specificity
        specificity = tn / (tn + fp)

        ## Positive predictive value
        positive_predictive_value = precision

        ## Negative predictive value
        negative_predictive_value = tn / (tn + fn)

        ## Balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2

        if auto_threshold:
            return area_under_curve, accuracy, f1_score, sensitivity, specificity, positive_predictive_value, negative_predictive_value, balanced_accuracy, prediction_threshold
        else:
            return area_under_curve, accuracy, f1_score, sensitivity, specificity, positive_predictive_value, negative_predictive_value, balanced_accuracy 
    else:
        return area_under_curve