from panda_physics import PandaGraspSimulation

def main():
    # Create and run simulation
    app = PandaGraspSimulation()
    
    # Run with command line arguments
    app.run()
    
    # Post-process results
    app.post_process(app.info_filename)

if __name__ == "__main__":
    main()