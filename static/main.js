console.log("Script loaded!");

// function uploadCSV() {
//   const fileInput = document.getElementById("csvFile");
//   const file = fileInput.files[0];
//   if (file) {
//     alert("File selected: " + file.name);
//   } else {
//     alert("Please select a file first.");
//   }
// }

function showToast() {
      const toast = document.getElementById("toast");
      toast.className = "show";

      // Hide after 3 seconds
      setTimeout(() => {
        toast.className = toast.className.replace("show", "");
      }, 3000);
    }

function submitForm() {
    const fileInput = document.getElementById('csvFile');
    if (fileInput.files.length === 0) {
        alert("Please choose a file before submitting.");
        return;
    }
    document.getElementById('csvForm').submit();
}

function uploadCSV() {
  const fileInput = document.getElementById("csvFile"); // Get the file input element
  const file = fileInput.files[0]; // Get the first selected file (the CSV)

  if (!file) { // If no file selected, alert user
    alert("Please select a file first.");
    return;
  }

  const reader = new FileReader(); // Create a new FileReader object

  reader.onload = function(event) {  // When file is loaded (read)
    const text = event.target.result; // The file content is here as text
    displayPreview(text);  // Call a function to show preview of CSV data
  };

  reader.readAsText(file);  // Read the file as plain text (CSV is text format)
}

function displayPreview(csvText) {
  const r = csvText.split("\n");
  const row_count = r.length -1;
  const rows = csvText.split("\n").slice(0, 11); // Split text into lines, take first 10 lines
  let html = "<table border='1' class='preview-table'><thead>";

  const headers = rows[0].split(","); // First line = headers, split by comma
  const column_count = headers.length;
  html += "<tr>" + headers.map(h => `<th>${h.trim()}</th>`).join("") + "</tr></thead><tbody>";

  // For each remaining line, split by comma and add table rows
  for (let i = 1; i < rows.length; i++) {
    const cells = rows[i].split(",");
    html += "<tr>" + cells.map(c => `<td>${c.trim()}</td>`).join("") + "</tr>";
  }
  html += "</tbody></table>";

  document.getElementById("preview").innerHTML = "<h2>Your Data.....</h2>";
  document.getElementById("preview").innerHTML += "<p>This is the first 10 data from your uploaded File.</p>";
  document.getElementById("preview").innerHTML += html;  // Show the table in the preview div
  document.getElementById("file_summery").style.display = "block";
  document.getElementById("file_summery").innerHTML = `<h3>File Summery...</h3>`;
  document.getElementById("file_summery").innerHTML += `<p>Number of Row : ${row_count}</p>`;
  document.getElementById("file_summery").innerHTML += `<p>Number of Column : ${column_count}</p>`;
  document.getElementById("file_summery").innerHTML += `<p>Columns : [${headers}]</p>`;

}


function showForm(formId) {
      const forms = document.querySelectorAll('.form-container');
      forms.forEach(form => form.style.display = 'none');

      document.getElementById(formId).style.display = 'block';
    }

function downloadAsCSV() {
            const table = document.querySelector("#normal-table table");
            if (!table) {
                alert("No normal table found.");
                return;
            }

            let csv = [];
            for (let row of table.rows) {
                let cols = Array.from(row.cells).map(cell => `"${cell.innerText.replace(/"/g, '""')}"`);
                csv.push(cols.join(","));
            }

            const csvContent = csv.join("\n");
            const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
            const link = document.createElement("a");
            const url = URL.createObjectURL(blob);

            link.setAttribute("href", url);
            link.setAttribute("download", "normal_table.csv");
            link.style.visibility = "hidden";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

