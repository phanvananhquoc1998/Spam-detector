// import axios from "axios";
const baseURL = "http://localhost:5000";
function init_table(options) {
  options = options || {};
  var csv_path = options.csv_path || "";
  var el = options.element || "table-container";
  var allow_download = options.allow_download || false;
  var csv_options = options.csv_options || {};
  var datatables_options = options.datatables_options || {};

  $(el).html(
    "<table class='table table-striped table-condensed' id='my-table' style='width: 100%;'></table>"
  );
  $.when($.get(csv_path)).then(function (data) {
    var csv_data = $.csv.toArrays(data, csv_options);
    var table_head = "<thead><tr>";
    for (let head_id = 0; head_id < csv_data[0].length; head_id++) {
      const val = csv_data[0][head_id];
      table_head += "<th>" + val + "</th>";
    }

    table_head += "</tr></thead>";

    var tbBody = "<tbody>";
    for (let row_id = 1; row_id < csv_data.length; row_id++) {
      tbBody += "<tr>";

      for (let col_id = 0; col_id < csv_data[row_id].length; col_id++) {
        const val = csv_data[row_id][col_id];
        tbBody += "<td>" + val + "</td>";
      }

      tbBody += "</tr>";
    }
    tbBody += "</tbody>";
    $("#my-table").append($(table_head));
    $("#my-table").append($(tbBody));

    $("#my-table").DataTable(datatables_options);

    if (allow_download)
      $("#" + el).append(
        "<p><a class='btn btn-info' href='" +
          csv_path +
          "'><i class='glyphicon glyphicon-download'></i> Download as CSV</a></p>"
      );
  });
}

function checkSection() {
  const pathname = window.location.pathname;
  if (pathname !== "/") {
    $("html, body").animate(
      {
        scrollTop: $(`#${pathname.slice(1)}_section`).offset().top,
      },
      2000
    );
  }
}

$(() => {
  console.log("TODO: train an execlent artificial intelligent!!");

  checkSection();

  $("#customFile").change((e) => {
    $('label[for="customFile"]').html(e.target.value);
  });

  $("#filename").change((e) => {
    $('label[for="filename"]').html(e.target.value);
  });

  const valSelected = $("#model").attr("value");
  $("#model").val(valSelected);
  var imgModel = valSelected + "-train";

  $("#model-preview img").attr(
    "src",
    `${baseURL}/static/images/${imgModel}.jpg`
  );

  $("#model").change((e) => {
    imgModel = "" + $(e.target).val() + "-train";
    $("#model-preview img").attr(
      "src",
      `${baseURL}/static/images/${imgModel}.jpg`
    );
  });

  // preprocessing step
  let arrLiStep = $(".preprocessing-step li");
  let arrTimeout = [700, 1900, 3000, 4700];
  var showStep = () => {
    arrLiStep.each((index, ele) => {
      setTimeout(() => {
        $(ele).removeClass("d-none");
        if (index > 0) {
          $(arrLiStep[index - 1]).addClass("done");
        }
        if (index === arrLiStep.length - 1) {
          setTimeout(() => {
            $(arrLiStep[index]).addClass("done");
            $("#connector-preprocessing").css({ display: "block" });
            $("#table_dataset").fadeIn("5000");
          }, 1000);
        }
      }, arrTimeout[index]);
    });
  };
  showStep();

  // change tab predict
  $('a[data-toggle="tab"]').on("shown.bs.tab", function (e) {
    var target = $(e.target).attr("href"); // activated tab
    $("#predictLabel").val("");
    $("#predictLabel").html("");
    $("#filename").val("");
    $('label[for="filename"]').html("Predict file");
    console.log(target);
    $("#predict_section")[0].reset();
  });

  // review dataset
  const reviewFile = (element, fileInput) => {
    $(element).click((e) => {
      $("#table-review").html("");
      const file = $(fileInput).prop("files")[0];
      const csvPath = (window.URL || window.webkitURL).createObjectURL(file);
      init_table({
        csv_path: csvPath,
        element: "#table-review",
        datatables_options: {
          paging: false,
          ordering: false,
          info: false,
          columnDefs: [
            { width: "20%", targets: 0 },
            { width: "120%", targets: 1 },
          ],
        },
      });
      $("#modalReview").modal("show");
    });
  };
  //   let filePredict = {{
  //     res['predict_result'] if res['predict_result'] != None and res['predict_label'] == '' else
  //   ''
  // }};
  reviewFile("#review", "#customFile");
  reviewFile("#reviewFileRes", "#filename");
});
