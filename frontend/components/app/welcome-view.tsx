import { Button } from '@/components/ui/button';

function WelcomeImage() {
  return (
    <svg
      width="64"
      height="64"
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="mb-4 size-16 text-teal-600 dark:text-teal-400"
    >
      <path
        d="M32 56C32 56 8 42.4 8 24.8C8 16.6 14.6 10 22.8 10C27 10 31 12 32 15.4C33 12 37 10 41.2 10C49.4 10 56 16.6 56 24.8C56 30.8 51.6 36.6 46 41.6"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M14 30H22L26 22L32 38L36 28L39 30H50"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: () => void;
}

export const WelcomeView = ({
  startButtonText,
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  return (
    <div ref={ref}>
      <section className="bg-background flex flex-col items-center justify-center px-6 text-center">
        <WelcomeImage />

        <h1 className="text-foreground text-xl font-semibold">Saathi</h1>

        <p className="text-foreground mt-2 max-w-xs pt-1 leading-6 font-medium">
          Apni health se juda koi bhi sawaal poochhein — Hindi, English, ya
          dono mila kar. Saathi sunega aur samjhayega.
        </p>

        <Button
          size="lg"
          onClick={onStartCall}
          className="mt-6 w-64 rounded-full bg-teal-600 font-mono text-xs font-bold tracking-wider text-white uppercase hover:bg-teal-700 dark:bg-teal-500 dark:hover:bg-teal-600"
        >
          {startButtonText}
        </Button>

        <p className="text-muted-foreground mt-4 max-w-xs text-xs leading-5">
          Saathi doctor nahi hai — sirf jaankari deta hai. Kisi bhi serious
          symptom ke liye turant doctor se milein.
        </p>
      </section>

      <div className="fixed bottom-5 left-0 flex w-full items-center justify-center px-4">
        <p className="text-muted-foreground max-w-prose pt-1 text-center text-xs leading-5 font-normal text-pretty md:text-sm">
          Built with Murf Falcon — part of{' '}
          <span className="font-medium">10 Days of Voice Agents</span>.
        </p>
      </div>
    </div>
  );
};