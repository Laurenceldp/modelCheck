<?php
//checks for user logged in 
/* session_start();
include("conn.php");

ob_start();
if (!isset($_SESSION['clientloggedin'])) {

    header('location:signup.php');
    ob_end_flush();
    $_SESSION['ClientID'];
} else {
    $user = $_SESSION['clientloggedin'];
}


// get Users
$clientdetails  = "SELECT * FROM TensorData WHERE UserID='$user';";
$resultcd = $conn->query($clientdetails);
if (!$resultcd) {
    echo "$conn->error";
}
$users = array();
$numrows = $resultcd->num_rows;
while ($rows = $resultcd->fetch_assoc()) {
    $users[] = $rows;
}

echo '<pre>';
print_r($users[0]);
echo '</pre>'; */
?>
<script src="js/tensorflow.js"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js"></script>
<!-- Import tfjs-vis -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@1.0.2/dist/tfjs-vis.umd.min.js"></script>
<script type="text/javascript" src="https://laurencefay.com/examdetector/model/examDetectorModel.json"></script>
<script
  src="https://code.jquery.com/jquery-3.5.1.min.js"
  integrity="sha256-9/aliU8dGd2tb6OSsuzixeV4y/faTqgFtohetphbbj0="
  crossorigin="anonymous"></script>
<script>
    
/*     let predictData = <?php //echo json_encode($users, JSON_NUMERIC_CHECK) ?>; */
    let predictData =  tf.data.csv("https://laurencefay.com/examdetector/data/46data.csv");

    console.log("Predict data on line 53:" + predictData.length);
    console.log("predict data " + predictData[0]);
    
    
/*  async function getFeatures(){
await fetch("https://laurencefay.com/examdetector/model/tValues.json")
        .then(function(resp) {
            return resp.json();
            console.log(resp.json());
        })
        .then(function(data) {
           normalisedFeatureJ = data.normalisedFeature;
            console.log(normalisedFeatureJ);
            console.log(normalisedFeatureJ.max);
return normalisedFeatureJ;
        });
   
}  */
let normalisedFeatureJ = {};
    $.ajax({
        url: "https://laurencefay.com/examdetector/model/tValuesx.json",
        async: false,
        dataType: 'json',
        success: function(data) {
            normalisedFeatureJ = (data);
        }
    });
console.log(Object.values(normalisedFeatureJ));
//const normTensor = normalisedFeatureJ.dataSync()[0];


//console.log(normTensor+ "is norm tensor");
//console.log(normalisedFeatureJ.tensor.min);
    // jsonIssues accessible here -- good!!
/* $.getJSON('https://laurencefay.com/examdetector/model/tValues.json', function(data) {
    console.log('data',data);
});
const normalisedFeatureJ = data;
//const normalisedFeature = await getFeatures(); */

//getFeatures();
    /* console.log(normalisedFeatureJ + "This is the normalised feature");
    console.log(normalisedFeatureJ.min+"is the normalied feature min");
    console.log(normalisedFeatureJ[0]+"is the normalised feature max"); */

    let minF = tensor({"min":{"isDisposedInternal":false,"shape":[],"dtype":"float32","size":1,"strides":[],"dataId":{},"id":6,"rankType":"0"}}),
     maxF =   tensor({"max":{"isDisposedInternal":false,"shape":[],"dtype":"float32","size":1,"strides":[],"dataId":{},"id":16,"rankType":"0"}});

    function normalise(tensor, previousMin = null, previousMax = null) {
        const min = previousMin || tensor.min();
        console.log("tensor min for normalised is :" + tensor.min());
        const max = previousMax || tensor.max();
        console.log("tensor max for normalised is :" + tensor.max());

        const normalisedTensor = tensor.sub(min).div(max.sub(min));
       // const normalisedTensor = (tensor-min)/(max-min);
        return {
            tensor: normalisedTensor,
            min,
            max
        };
    }

    function denormalise(tensor, min, max) {
        console.log("tensor min for denormalised is :" + min);
        console.log("tensor max for denormalised is :" + max);
        const denormalisedTensor = tensor.mul(max.sub(min)).add(min);
        return denormalisedTensor;
    }

    let pointsPredict;
    let inputTensor, normalisedInput, outPutCheck, outputValue = [];
    let cheatArray = [];
    async function predict() {
        const model = await tf.loadLayersModel('https://laurencefay.com/examdetector/model/examDetectorModel.json');
        tfvis.show.modelSummary({
            name: "Model summary"
        }, model);
        const layer = model.getLayer(undefined, 0);
        tfvis.show.layer({
            name: "Layer 1"
        }, layer);

        console.log("Predict data on line 79:" + predictData.length);
        console.log("predict data " + predictData[0]);
        await predictData.forEach(e => cheatArray.push(e));
        console.log("cheat array is: " + cheatArray);
        console.table(cheatArray[0]);
        // Extract x and y values to plot
        predictPointsDataset = ((cheatArray.map(record => ({
            x: record.Ntxaxis,
            x1: record.Ntyaxis,
            x2: record.Ntzaxis,
            x3: record.Lcxaxis,
            x4: record.Lcyaxis,
            x5: record.Lczaxis,
            x6: record.Rcxaxis,
            x7: record.Rcyaxis,
            x8: record.Rczaxis,
            x9: record.Mexaxis,
            x10: record.Meyaxis,
            x11: record.Mezaxis,
            x12: record.FacesCount,
            x13: record.Direction,
            x14: record.VisibleCount,
            x15: record.FirstObj,
            x16: record.SecondObj,



        }))));
        console.log("predict points dataset: " + predictPointsDataset);
        pointsPredict = await predictPointsDataset;
        //   tf.tidy(() => {
        console.log(pointsPredict.length + "is points predict length");

        console.log("is points predict: " + pointsPredict);

        const featureValuesPredict = pointsPredict.map(p => [p.x, p.x1, p.x2, p.x3, p.x4, p.x5, p.x6,
            p.x7, p.x8, p.x9, p.x10, p.x11, p.x12, p.x13, p.x14, p.x15, p.x16
        ]);

        console.log(featureValuesPredict.length + "is feature predict length");
        console.log("This is line 155 - featureValuesPredict" + featureValuesPredict);

        console.log(typeof featureValuesPredict[0][1]);

        const inputTensor = tf.tensor(featureValuesPredict);


        console.log(inputTensor + "This is line 159 - inputTensor");



        const normalisedInput = normalise(inputTensor, minF, maxF);
        console.log(normalisedInput.tensor + "this is the normalised input on line 160");
        console.log(normalisedInput);

        const outPutCheck = model.predict(normalisedInput.tensor);
        console.log(outPutCheck + ": this is the output on line 164");

        const outputTensor = denormalise(outPutCheck);

        console.log("output tensor is :" + outputTensor);

        let outputFloat = outputTensor.dataSync();
        console.log("outputfloat is : " + outputFloat);
        let arrayOutput = [];
        for (let i = 0; i < outputFloat.length; i++) {
            arrayOutput.push(outputFloat[i]);
        }

        console.log(arrayOutput + "is arrayOutput after push")
        //Work out the sum of the numbers in
        //our array
        let totalSum = 0;
        for (let i = 0; i < arrayOutput.length; i++) {
            totalSum += arrayOutput[i];
        }

        console.log(totalSum + ": is total sum")
        //Work out how many numbers are
        //in our array.
        let sumCount = arrayOutput.length;

        console.log(sumCount + ": is length")
        //Finally, get the average.
        let ave = totalSum / sumCount;
        averagePercent = ave.toFixed(2) * 100;
        //Print the median / average to the console.
        //In this case, the average is 7.
        console.log("Prediction on cheating is: " + averagePercent + "%");

        /*    document.getElementById("prediction-output").innerHTML = `The predicted % of the likely hood of cheating is <br>` +
               `<span style="font-size: 2em">${averagePercent}\%</span>`; */
    };


    predict();
</script>
