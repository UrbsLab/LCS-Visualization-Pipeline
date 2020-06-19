import React from "react";
import {
  Dialog,
  TextField,
  Button,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Input,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  Menu,
} from "@material-ui/core";

class ExperimentDialog extends React.Component {
  render() {
    return (
      <Dialog
        open={this.props.shouldDisplay}
        onClose={this.props.close}
        aria-labelledby="form-dialog-title"
      >
        <DialogTitle id="form-dialog-title">Add a new experiment</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            id="expname"
            label="Experiment Name"
            type="text"
            fullWidth
            style={{ margin: 10 }}
          />
          <Input type="file" style={{ display: "block", margin: 10 }} />
          <FormControl style={{ minWidth: 120, margin: 10 }}>
            <InputLabel>Class Label</InputLabel>
            <Select>
              <MenuItem value="Class">
                <em>Class</em>
              </MenuItem>
              <MenuItem value="Feature1">
                <em>None</em>
              </MenuItem>
            </Select>
          </FormControl>
          <FormControl style={{ minWidth: 150, margin: 10 }}>
            <InputLabel>Instance Label</InputLabel>
            <Select>
              <MenuItem value="Class">
                <em>Instance</em>
              </MenuItem>
              <MenuItem value="Feature1">
                <em>None</em>
              </MenuItem>
            </Select>
          </FormControl>
          <FormControl style={{ minWidth: 120, margin: 10 }}>
            <InputLabel>Group Label</InputLabel>
            <Select>
              <MenuItem value="Class">
                <em>Group</em>
              </MenuItem>
              <MenuItem value="Feature1">
                <em>None</em>
              </MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={this.props.addExperiment} color="primary">
            Add Experiment
          </Button>
        </DialogActions>
      </Dialog>
    );
  }
}

export default ExperimentDialog;
