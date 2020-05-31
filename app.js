var fs = require("fs");
const axios = require("axios").default;

function download(date, writeDir = "./output/") {
  const writer = fs.createWriteStream(`${writeDir}${date}.jpg`);

  return axios({
    method: "get",
    url: `http://creodias.sentinel-hub.com/ogc/wms/56d1742b-1dd7-4ad4-9d20-7eb08b2744a3?version=1.1.1&service=WMS&request=GetMap&format=image%2Fjpeg&srs=EPSG%3A3857&layers=NO2_VISUALIZED&bbox=-2397066%2C4109254%2C1115369%2C5699144&time=2019-01-01T00%3A00%3A00Z%2F${date}T23%3A59%3A59Z&width=1456&height=650&showlogo=true&nicename=Sentinel-2%20L1C%20image%20on%202019-07-18.jpg&bgcolor=000000&maxcc=20&evalsource=S2`,
    responseType: "stream",
  })
    .then((res) => {
      return new Promise((resolve, reject) => {
        res.data.pipe(writer);
        let error = null;

        writer.on("error", (err) => {
          error = err;
          writer.close();
          reject(err);
        });

        writer.on("close", () => {
          if (!error) {
            resolve(`Successfully downloaded image for ${date}`);
          }
        });
      });
    })
    .then((success) => console.log(success))
    .catch((err) => {
      // handle timeouts  and internal server error -- try again
      if (
        err.response &&
        (err.response.status === 504 || err.response.status === 500)
      ) {
        console.log(`Timed out; retrying for ${date}`);
        return download(date, writeDir);
      }
      console.log(err);
    });
}

function run() {
  var dates = [];
  var now = new Date();

  // populate dates array so that we can call all of them in in parallel with promise.all
  for (var d = new Date(2020, 0, 1); d <= now; d.setDate(d.getDate() + 1)) {
    // need to be in YYYY-MM-DD
    let yyyy = String(d.getFullYear());

    let mm = String(d.getMonth() + 1);
    if (mm.length < 2) {
      mm = "0" + mm;
    }

    let dd = String(d.getDate());
    if (dd.length < 2) {
      dd = "0" + dd;
    }

    let formattedDate = [yyyy, mm, dd].join("-");
    dates.push(formattedDate);
  }

  // downloads them all in parallel
  return Promise.all(dates.map((date) => download(date)));
}

run();
