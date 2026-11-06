"""Serve a read-only local-network map for viewing the Rubik Pi from a browser."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading


class BrowserMap:
    """Publish a thread-safe map snapshot without exposing files or robot controls."""

    def __init__(self, host='0.0.0.0', port=8080):
        """Start the lightweight HTTP viewer on the robot's network interface."""
        self.lock = threading.Lock()
        self.data = {'status': 'waiting for estimates', 'trajectory': [], 'landmarks': [], 'pose': None}
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner._respond(self)
            def log_message(self, format, *args):
                pass
        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def _respond(self, request):
        """Expose only the viewer HTML and current snapshot, never arbitrary filesystem paths."""
        path = request.path.split('?', 1)[0]
        if path == '/':
            payload, content_type = PAGE.encode(), 'text/html; charset=utf-8'
        elif path == '/state.json':
            with self.lock:
                payload = json.dumps(self.data, allow_nan=False).encode()
            content_type = 'application/json'
        else:
            request.send_error(404)
            return
        request.send_response(200)
        request.send_header('Content-Type', content_type)
        request.send_header('Content-Length', str(len(payload)))
        request.send_header('Cache-Control', 'no-store')
        request.end_headers()
        try:
            request.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def update(self, estimator, stamp, status, lap, vision_age, skipped_frames):
        """Append the estimated trajectory and replace the displayed map/covariance snapshot."""
        landmarks = []
        for key in estimator.landmark_ids:
            position, covariance = estimator.landmark(key)
            landmarks.append({'id': f'{key[0]}:{key[1]}', 'position': position.tolist(),
                              'covariance': covariance.tolist()})
        pose = estimator.robot_pose.tolist()
        with self.lock:
            trajectory = self.data['trajectory'] + [[stamp, *pose]]
            self.data = {'stamp': stamp, 'status': status, 'pose': pose,
                         'trajectory': trajectory[-10000:], 'landmarks': landmarks,
                         'lap': lap, 'vision_age_s': vision_age, 'skipped_frames': skipped_frames}

    def close(self):
        """Stop the HTTP listener and join its worker before the ROS node exits."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.)


PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HW3 live robot map</title>
<style>
body{margin:0;background:#f4f6f9;color:#172033;font:15px system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:24px}h1{font-size:25px;margin:0 0 8px}
header{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}
button{padding:9px 15px;border:1px solid #bbc6d4;border-radius:7px;background:white;cursor:pointer}
#status{padding:12px;background:white;border:1px solid #dbe2eb;border-radius:8px;margin:14px 0}
#legend{margin:10px 0;color:#526078}canvas{width:100%;height:64vh;min-height:350px;background:white;border:1px solid #dbe2eb;border-radius:10px}
.blue{color:#176cce}.green{color:#218547}table{width:100%;border-collapse:collapse;margin-top:16px;background:white}
th,td{text-align:left;padding:9px;border-bottom:1px solid #e5eaf0;font-variant-numeric:tabular-nums}
small{color:#526078}#connection{font-weight:600} .warning{color:#ad4c00}
</style><main>
<header><div><h1>HW3 live robot map</h1><small>Estimated positions in the local starting frame · meters</small></div>
<div><span id="connection">Connecting…</span> <button id="follow">Pause display</button></div></header>
<div id="status">Waiting for the robot…</div>
<div id="legend"><span class="blue">● Estimated trajectory and robot heading</span> ·
<span class="green">● Landmarks and 95% covariance ellipses</span> · gray grid: meters</div>
<canvas id="map" aria-label="Live estimated robot trajectory and landmark map"></canvas>
<small>Display-only page. Pause display does not stop the robot. Without tag observations, the blue path is command dead reckoning; no landmark positions are invented.</small>
<table><thead><tr><th>Landmark</th><th>x (m)</th><th>y (m)</th><th>σx (m)</th><th>σy (m)</th></tr></thead><tbody id="landmarks"></tbody></table>
</main><script>
const canvas=document.getElementById('map'), ctx=canvas.getContext('2d');
const connection=document.getElementById('connection'), status=document.getElementById('status');
let state=null, paused=false, received=0;
document.getElementById('follow').onclick=event=>{paused=!paused;event.target.textContent=paused?'Resume display':'Pause display';};
function ellipse(landmark){
 const c=landmark.covariance,a=c[0][0],b=(c[0][1]+c[1][0])/2,d=c[1][1];
 const delta=Math.hypot(a-d,2*b),theta=.5*Math.atan2(2*b,a-d);
 const r1=Math.sqrt(Math.max(0,(a+d+delta)/2)*5.991464547),r2=Math.sqrt(Math.max(0,(a+d-delta)/2)*5.991464547);
 return Array.from({length:65},(_,i)=>{let t=i*Math.PI/32,u=r1*Math.cos(t),v=r2*Math.sin(t);
 return [landmark.position[0]+u*Math.cos(theta)-v*Math.sin(theta),landmark.position[1]+u*Math.sin(theta)+v*Math.cos(theta)];});
}
function draw(data){
 const ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;
 canvas.width=w*ratio;canvas.height=h*ratio;ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
 const path=data.trajectory.map(p=>[p[1],p[2]]), ellipses=data.landmarks.map(ellipse);
 const points=[[0,0],[2.1336,2.1336],...path,...data.landmarks.map(l=>l.position),...ellipses.flat()];
 let xmin=Math.min(...points.map(p=>p[0]))-.3,xmax=Math.max(...points.map(p=>p[0]))+.3;
 let ymin=Math.min(...points.map(p=>p[1]))-.3,ymax=Math.max(...points.map(p=>p[1]))+.3;
 const scale=Math.min((w-100)/(xmax-xmin),(h-80)/(ymax-ymin));
 const xmid=(xmin+xmax)/2,ymid=(ymin+ymax)/2;
 function xy(p){return [w/2+(p[0]-xmid)*scale,h/2-(p[1]-ymid)*scale];}
 function line(points,color,width){if(!points.length)return;ctx.beginPath();points.forEach((p,i)=>{const [x,y]=xy(p);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
 ctx.font='12px system-ui';ctx.fillStyle='#66758b';
 const step=Math.max(.5,Math.pow(10,Math.floor(Math.log10(Math.max(xmax-xmin,ymax-ymin)/10))));
 for(let x=Math.ceil(xmin/step)*step;x<=xmax;x+=step){line([[x,ymin],[x,ymax]],'#e8edf3',1);const [px]=xy([x,ymin]);ctx.fillText(x.toFixed(1),px,h-12);}
 for(let y=Math.ceil(ymin/step)*step;y<=ymax;y+=step){line([[xmin,y],[xmax,y]],'#e8edf3',1);const [,py]=xy([xmin,y]);ctx.fillText(y.toFixed(1),10,py);}
 line(path,'#176cce',2.5);
 data.landmarks.forEach((l,i)=>{line(ellipses[i],'#218547',1.5);const [x,y]=xy(l.position);ctx.fillStyle='#218547';ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);ctx.fill();ctx.fillText(l.id,x+8,y-8);});
 if(data.pose){const [x,y,yaw]=data.pose;line([[x,y],[x+.2*Math.cos(yaw),y+.2*Math.sin(yaw)]],'#176cce',4);const p=xy([x,y]);ctx.fillStyle='#176cce';ctx.beginPath();ctx.arc(...p,5,0,2*Math.PI);ctx.fill();}
 const age=data.vision_age_s==null?'no accepted tags yet':`last accepted tag ${data.vision_age_s.toFixed(1)}s ago`;
 status.textContent=`${data.status} · lap ${data.lap||0} · ${data.landmarks.length} landmarks · ${age}`;
 const table=document.getElementById('landmarks');table.replaceChildren();
 data.landmarks.forEach(l=>{const row=document.createElement('tr');[l.id,...l.position.map(x=>x.toFixed(3)),Math.sqrt(Math.max(0,l.covariance[0][0])).toFixed(3),Math.sqrt(Math.max(0,l.covariance[1][1])).toFixed(3)].forEach(text=>{const cell=document.createElement('td');cell.textContent=text;row.appendChild(cell);});table.appendChild(row);});
}
async function poll(){
 try{const response=await fetch('/state.json',{cache:'no-store'});if(!response.ok)throw Error('HTTP error');const data=await response.json();
 if(data.stamp!==received){received=data.stamp;window.lastChange=Date.now();}
 connection.textContent=window.lastChange&&Date.now()-window.lastChange>3000?'Estimates stale':'Connected';
 connection.className=connection.textContent==='Connected'?'':'warning';
 if(!paused){state=data;draw(state);}}
 catch(error){connection.textContent='Disconnected — retrying';connection.className='warning';}
 finally{setTimeout(poll,500);}
}
window.addEventListener('resize',()=>{if(state)draw(state);});poll();
</script></html>'''
